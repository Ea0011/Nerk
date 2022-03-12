from networks.losses import PerceptualLossVgg
from networks.utils import construct_unet
from processing.transforms import OutputTransform
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchvision
from collections import OrderedDict
from kornia.geometry.transform import get_tps_transform, warp_image_tps

class DiscriminatorModule(nn.Module):
  def __init__(self, discriminator_params):
    super().__init__()
    self.discriminator_params = discriminator_params
    self.discriminator, self.discriminator_bottleneck = construct_unet(self.discriminator_params)
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(discriminator_params['encoder_blocks'][-1]['out_c'], 1),
      nn.Sigmoid(),
    )
  
  def forward(self, input):
    for _, layer in enumerate(self.discriminator):
      _, input = layer(input)

    for _, layer in enumerate(self.discriminator_bottleneck):
      input = layer(input)

    validity = self.classifier(input)

    return validity

# TODO: make texture transfer perform multiple attention + convolution operations
class TextureTransfer(nn.Module):
  def __init__(self, attn_heads, in_dim=256) -> None:
    super().__init__()
    self.attention = nn.MultiheadAttention(embed_dim=in_dim, num_heads=attn_heads, batch_first=True)

  def forward(self, sketch, exemplar):
    attn_out, _ = self.attention(sketch, exemplar, exemplar)
    texture_transfer = torch.cat((sketch, attn_out), dim=2)

    return texture_transfer


class SketchColorizer(nn.Module):
  def __init__(self, colorizer_params, style_params):
    super().__init__()
    self.colorizer_params = colorizer_params
    self.colorizer_encoder, self.colorizer_bottleneck, self.colorizer_decoder, self.colorizer_output = construct_unet(colorizer_params)
    self.style_encoder, self.style_bottleneck = construct_unet(style_params)
    self.texture_transfer = TextureTransfer(8, self.colorizer_params['encoder_blocks'][-1]['out_c']) # TODO: parametrize

  def forward(self, sketch, exemplars):
    skip = []

    for i, layer in enumerate(self.colorizer_encoder):
      s, sketch = layer(sketch)
      skip.append(s)

    for i, layer in enumerate(self.colorizer_bottleneck):
      sketch = layer(sketch)

    for i, layer in enumerate(self.style_encoder):
      _, exemplars = layer(exemplars)

    for i, layer in enumerate(self.style_bottleneck):
      exemplars = layer(exemplars)

    # B x C x H x W -> B x H*W x C
    # Compute attention to transfer texture from exemplar to sketch
    b, c, h, w = sketch.shape
    transfer = self.texture_transfer(sketch.view((b, h*w, c)), exemplars.view((b, h*w, c)))
    sketch = transfer.view((b, c*2, h, w))

    for i, layer in enumerate(self.colorizer_decoder):
      sketch = layer(sketch, skip[-(i + 1)])

    for i, layer in enumerate(self.colorizer_output):
      sketch = layer(sketch)

    return sketch

class SketchColoringModule(pl.LightningModule):
  r'''
  A module for image colorization.
  Hparams structure
  colorizer_params: 
    {
      'encoder_blocks': [{'in_c', 'out_c'}]
      'decoder_blocks': [{'in_c', 'out_c'}]
    }
  discriminator_params: 
    {
      'encoder_blocks': [{'in_c', 'out_c'}]
    }
  style_params: 
    {
      'encoder_blocks': [{'in_c', 'out_c'}]
    }
  num_exemplars: int
  train_gan: boolean
  g: int
  rec: int
  perc: int
  colorizer_lr: float
  discriminator_lr: float
  b1: float
  b2: float
  weight_decay: float
  device: torch.device
  perceptual_layer: int, layer number of VGG network
  '''
  def __init__(self, device, **hparams):
    super(SketchColoringModule, self).__init__()

    # save network hyperparameters to checkpoints
    self.save_hyperparameters()

    # Used to log compute graph to tensorboard
    self.example_input_array = [
      torch.zeros((1, 1, 512, 512), dtype=torch.float), # sktech
      torch.zeros((1, 2, 512, 512), dtype=torch.float), # exemplar
    ]

    # initialize generator and dsicriminator
    self.Generator = SketchColorizer(self.hparams.colorizer_params, self.hparams.style_params)

    if self.hparams.train_gan:
      self.Discriminator = DiscriminatorModule(self.hparams.discriminator_params)

    # Initialiaze colored sketch generator
    self.reconstruction_loss = nn.SmoothL1Loss(beta=0.1)
    self.adversarial_loss = nn.BCELoss()
    self.perceptual_loss = PerceptualLossVgg(device=device, layer=self.hparams.perceptual_layer)
    self.lab_to_rgb = OutputTransform()
    
    # initialize a variable to hold generated images
    self.generated_imgs = None

    # Maybe some loss with high level VGG layers
    # self.content_loss

  def forward(self, sketches, exemplars):
    colored = self.Generator(sketches, exemplars)
    return colored

  def training_step(self, batch, batch_idx, optimizer_idx=0):
    images, sketches, exemplars, images_lab = self._exemplars_from_batch(batch)

    # train generator
    if optimizer_idx == 0:
      # generate images
      self.generated_imgs = self(sketches, exemplars)

      # log sampled images
      sample_imgs = self.generated_imgs[:6]
      sample_imgs = self.lab_to_rgb(sample_imgs)
      grid = torchvision.utils.make_grid(sample_imgs)
      self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

      g_loss = 0
      if self.hparams.train_gan:
        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(sketches.size(0), 1)
        valid = valid.type_as(sketches)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.Discriminator(self.generated_imgs), valid)

      color_channels = self.generated_imgs[:, 1:]
      rec_loss = self.reconstruction_loss(color_channels, images_lab[:, 1:])

      rgb_images = self.lab_to_rgb(self.generated_imgs)
      perc_loss = self.perceptual_loss(rgb_images, images)

      tqdm_dict = {"g_loss": g_loss, 'rec_loss': rec_loss}
      total_loss = self.hparams.g * g_loss + self.hparams.rec * rec_loss + self.hparams.perc * perc_loss
      output = OrderedDict({
        "loss": total_loss,
        "perceptual_loss": perc_loss,
        "reconstruction_loss": rec_loss,
        "progress_bar": tqdm_dict, 
        "log": tqdm_dict,
      })
      self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

      return output

    # train discriminator
    if optimizer_idx == 1:
      # Don't train with Adversarial loss if not needed
      if not self.hparams.train_gan:
        pass
      # Measure discriminator's ability to classify real from generated samples

      # how well can it label as real?
      valid = torch.ones(images.size(0), 0)
      valid = valid.type_as(images)

      real_loss = self.adversarial_loss(self.Discriminator(images), valid)

      # how well can it label as fake?
      fake = torch.zeros(images.size(0), 1)
      fake = fake.type_as(images)

      fake_loss = self.adversarial_loss(self.Discriminator(self(sketches, exemplars).detach()), fake)

      # discriminator loss is the average of these
      d_loss = (real_loss + fake_loss) / 2
      tqdm_dict = {"d_loss": d_loss}
      output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
      return output

  def validation_step(self, batch, batch_idx):
    images, sketches, exemplars, images_lab = self._exemplars_from_batch(batch)
    generated_images = self(sketches, exemplars)

    g_loss = 0
    if self.hparams.train_gan:
      valid = torch.ones(sketches.size(0), 1)
      valid = valid.type_as(sketches)
      g_loss = self.adversarial_loss(self.Discriminator(generated_images), valid)

    color_channels = generated_images[:, 1:]
    rec_loss = self.reconstruction_loss(color_channels, images_lab[:, 1:])

    rgb_images = self.lab_to_rgb(generated_images)
    perc_loss = self.perceptual_loss(rgb_images, images)

    total_loss = self.hparams.g * g_loss + self.hparams.rec * rec_loss + self.hparams.perc * perc_loss

    self.log('val_loss', total_loss, prog_bar=True)

    return total_loss

  def configure_optimizers(self):
    colorizer_lr = self.hparams.colorizer_lr
    b1 = self.hparams.b1
    b2 = self.hparams.b2
    opt_g = torch.optim.AdamW(
      self.Generator.parameters(),
      lr=colorizer_lr,
      betas=(b1, b2),
      weight_decay=self.hparams.weight_decay,)

    if self.hparams.train_gan:
      discriminator_lr = self.hparams.discriminator_lr
      opt_d = torch.optim.AdamW(self.Discriminator.parameters(), lr=discriminator_lr, betas=(b1, b2), weight_decay=self.hparams.weight_decay)

      return [opt_g, opt_d], []

    return [opt_g], []
  
  # An exemplar is an image from the same batch as the training sample
  def _exemplars_from_batch(self, batch):
    images, sketches, images_lab = batch
    images = torch.repeat_interleave(images, self.hparams.num_exemplars, dim=0)
    images_lab = torch.repeat_interleave(images_lab, self.hparams.num_exemplars, dim=0)
    sketches = torch.repeat_interleave(sketches, self.hparams.num_exemplars, dim=0)
    perm = torch.randperm(images.shape[0])
    exemplars = images_lab[perm, 1:] # retain only color information for exemplars

    return (images, sketches, exemplars, images_lab)

  def _exemplars_from_transformations(self, images):
    points_src = torch.rand(images.shape[0], 5, 2)
    points_dst = torch.rand(images.shape[0], 5, 2)
    # note that we are getting the reverse transform: dst -> src
    kernel_weights, affine_weights = get_tps_transform(points_dst, points_src)
    exemplars = warp_image_tps(images, points_src, kernel_weights, affine_weights)

    return exemplars
