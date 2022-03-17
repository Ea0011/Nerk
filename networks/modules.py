from networks.layers import VisualAttention
from networks.losses import PerceptualLossVgg
from networks.utils import construct_unet
from processing.transforms import DenormalizeLABImage, DenormalizeRGBImage, OutputTransform
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
    out_dim = 2 * self.discriminator_params['encoder_blocks'][-1]['out_c']
    linear_dim = self.discriminator_params['linear_dim']
    self.classifier = nn.Sequential(
      nn.Conv2d(out_dim, out_dim, kernel_size=linear_dim, padding=0),
    )
  
  def forward(self, input):
    for _, layer in enumerate(self.discriminator):
      _, input = layer(input)

    for _, layer in enumerate(self.discriminator_bottleneck):
      input = layer(input)

    validity = self.classifier(input)

    return validity

class TextureTransfer(nn.Module):
  def __init__(self, attn_dim, in_dim=512) -> None:
    super().__init__()
    self.attention = VisualAttention(in_dim=in_dim, attn_dim=attn_dim)
    self.normalization = nn.InstanceNorm2d(in_dim, affine=True)

  def forward(self, sketch, exemplar):
    texture_transfer, texture, attn_output_weights = self.attention(sketch, exemplar)
    texture_transfer = self.normalization(texture_transfer)

    return texture_transfer, texture, attn_output_weights


class SketchColorizer(nn.Module):
  def __init__(self, colorizer_params, style_params):
    super().__init__()
    self.colorizer_params = colorizer_params
    self.colorizer_encoder, self.colorizer_bottleneck, self.colorizer_decoder, self.colorizer_output = construct_unet(colorizer_params)
    self.style_encoder, self.style_bottleneck = construct_unet(style_params)
    self.texture_transfer = TextureTransfer(8, 2 * self.colorizer_params['encoder_blocks'][-1]['out_c'])

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

    # Compute attention to transfer texture from exemplar to sketch
    sketch, texture, attn_out_weights = self.texture_transfer(sketch, exemplars) # TODO: Visualize attention layer

    for i, layer in enumerate(self.colorizer_decoder):
      sketch, texture = layer(sketch, skip[-(i + 1)], texture)

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
  min_lr: float
  max_lr: float
  discriminator_lr: float
  b1: float
  b2: float
  weight_decay: float
  device: torch.device
  perceptual_layer: int, layer number of VGG network
  exemplar_method: string
  generator_frequency: int
  discriminator_frequency: int
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
    self.Discriminator = DiscriminatorModule(self.hparams.discriminator_params)

    # Initialiaze colored sketch generator
    self.color_loss = nn.SmoothL1Loss()
    self.reconstruction_loss = nn.SmoothL1Loss()
    self.adversarial_loss = nn.BCEWithLogitsLoss()
    self.discirminator_loss = nn.BCEWithLogitsLoss()
    self.perceptual_loss = PerceptualLossVgg(device=device, layer=self.hparams.perceptual_layer)
    self.lab_to_rgb = OutputTransform()
    self.denormalize_lab = DenormalizeLABImage()
    self.denormalize_rgb = DenormalizeRGBImage()
    
    # initialize a variable to hold generated images
    self.generated_imgs = None

    # Maybe some loss with high level VGG layers
    # self.content_loss

  def forward(self, sketches, exemplars):
    colored = self.Generator(sketches, exemplars)
    return colored

  def training_step(self, batch, batch_idx, optimizer_idx):
    images, sketches, exemplars, images_lab = self._exemplars_from_transformations(batch) if \
      self.hparams.exemplar_method == "self" else self._exemplars_from_batch(batch)

    # train generator
    if optimizer_idx == 0:
      # generate images
      self.generated_imgs = self(sketches, exemplars)

      # log sampled images
      sample_imgs = self.generated_imgs[:6].detach().clone()
      sample_imgs = self.lab_to_rgb(sample_imgs)
      grid = torchvision.utils.make_grid(sample_imgs)
      self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

      color_loss = self.color_loss(self.denormalize_lab(self.generated_imgs.clone())[:, 1:],
        self.denormalize_lab(images_lab.clone())[:, 1:])

      rgb_images = self.lab_to_rgb(self.generated_imgs.clone())

      perc_loss = self.perceptual_loss(rgb_images, images) if self.hparams.perc > 0 else 0
      rec_loss = self.reconstruction_loss(self.denormalize_rgb(rgb_images.clone()),
        self.denormalize_rgb(images.clone()))

      g_loss = 0
      if self.hparams.train_gan:
        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(sketches.size(0), 1)
        valid = valid.type_as(sketches)

        # adversarial loss is binary cross-entropy
        cgan_input = torch.cat([sketches, rgb_images], axis=1)
        g_loss = self.adversarial_loss(self.Discriminator(cgan_input), valid) / 2

      total_loss = self.hparams.g * g_loss + self.hparams.rec * rec_loss + self.hparams.perc * perc_loss + \
        self.hparams.color * color_loss

      generator_log = {"g_loss": g_loss, 'rec_loss': rec_loss, 'prec_loss': perc_loss, 'total_loss': total_loss, 'color_loss': color_loss} 

      output = OrderedDict({
        "loss": total_loss,
        "perceptual_loss": perc_loss,
        "reconstruction_loss": rec_loss,
        "progress_bar": generator_log, 
        "log": generator_log,
      })
      self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
      self.log_dict(generator_log, on_step=True, on_epoch=True, prog_bar=True, logger=True)

      return output

    # train discriminator
    if optimizer_idx == 1:
      # Don't train with Adversarial loss if not needed
      if not self.hparams.train_gan:
        pass
      # Measure discriminator's ability to classify real from generated samples

      # how well can it label as real?
      valid = torch.ones(images.size(0), 1)
      valid = valid.type_as(images)
      
      cgan_input = torch.cat([sketches, images], axis=1)
      real_loss = self.discirminator_loss(self.Discriminator(cgan_input), valid) / 2

      # how well can it label as fake?
      fake = torch.zeros(images.size(0), 1)
      fake = fake.type_as(images)

      self.generated_imgs = self(sketches, exemplars).detach()
      rgb_images = self.lab_to_rgb(self.generated_imgs)

      cgan_input = torch.cat([sketches, rgb_images], axis=1)
      fake_loss = self.discirminator_loss(self.Discriminator(cgan_input), fake) / 2

      # discriminator loss is the average of these
      d_loss = (real_loss + fake_loss) / 2
      discriminator_log = {"d_loss": d_loss, 'd_loss_fake': fake_loss, 'd_loss_real': real_loss}
      output = OrderedDict({"loss": d_loss, "progress_bar": discriminator_log, "log": discriminator_log})
      self.log_dict(discriminator_log, on_step=True, on_epoch=True, prog_bar=True, logger=True)
      return output

  def validation_step(self, batch, batch_idx):
    images, sketches, exemplars, images_lab = self._exemplars_from_transformations(batch) if \
      self.hparams.exemplar_method == "self" else self._exemplars_from_batch(batch)

    generated_images = self(sketches, exemplars)
    rgb_images = self.lab_to_rgb(generated_images.clone())

    g_loss = 0
    if self.hparams.train_gan:
      valid = torch.ones(sketches.size(0), 1)
      valid = valid.type_as(sketches)

      cgan_input = torch.cat([sketches, rgb_images], axis=1)
      g_loss = self.adversarial_loss(self.Discriminator(cgan_input), valid) / 2

    rec_loss = self.reconstruction_loss(self.denormalize_lab(generated_images.clone()), 
      self.denormalize_lab(images_lab.clone()))

    perc_loss = self.perceptual_loss(rgb_images, images) if self.hparams.perc > 0 else 0
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

    generator_scheduler = {
      'scheduler': torch.optim.lr_scheduler.CyclicLR(opt_g, base_lr=self.hparams.min_lr, max_lr=self.hparams.max_lr, step_size_up=50, cycle_momentum=False),
      'interval': 'step',
    }

    if self.hparams.train_gan:
      discriminator_lr = self.hparams.discriminator_lr
      opt_d = torch.optim.AdamW(self.Discriminator.parameters(), lr=discriminator_lr, betas=(b1, b2), weight_decay=self.hparams.weight_decay)

      discriminator_scheduler = {
        'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=100),
        'interval': 'epoch',
      }

      return (
        {
          'optimizer': opt_g,
          'lr_scheduler': generator_scheduler,
          'frequency': self.hparams.generator_frequency,
        },
        {
          'optimizer': opt_d,
          'lr_scheduler': discriminator_scheduler,
          'frequency': self.hparams.discriminator_frequency,
        },
      )

    return [opt_g], [generator_scheduler]
  
  # An exemplar is an image from the same batch as the training sample
  def _exemplars_from_batch(self, batch):
    images, sketches, images_lab = batch
    images = torch.repeat_interleave(images, self.hparams.num_exemplars, dim=0)
    images_lab = torch.repeat_interleave(images_lab, self.hparams.num_exemplars, dim=0)
    sketches = torch.repeat_interleave(sketches, self.hparams.num_exemplars, dim=0)
    perm = torch.randperm(images.shape[0])
    exemplars = images_lab[perm, 1:].clone() # retain only color information for exemplars

    return (images, sketches, exemplars, images_lab)

  def _exemplars_from_transformations(self, batch):
    images, sketches, images_lab = batch
    images = torch.repeat_interleave(images, self.hparams.num_exemplars, dim=0)
    images_lab = torch.repeat_interleave(images_lab, self.hparams.num_exemplars, dim=0)
    sketches = torch.repeat_interleave(sketches, self.hparams.num_exemplars, dim=0)

    xs = torch.tensor([-1.0, 1, 0, -1, 1])
    ys = torch.tensor([-1.0, -1, 0, 1, 1])

    points_src = torch.stack((xs, ys), 1).unsqueeze(0)
    points_src = torch.repeat_interleave(points_src, images_lab.shape[0], dim=0).type_as(images_lab)
    points_dst = points_src + torch.rand(images_lab.shape[0], 5, 2).type_as(images_lab) * 2 - .5
    # note that we are getting the reverse transform: dst -> src
    kernel_weights, affine_weights = get_tps_transform(points_dst, points_src)
    exemplars = warp_image_tps(images_lab.clone(), points_src, kernel_weights, affine_weights)

    return (images, sketches, exemplars[:, 1:], images_lab)