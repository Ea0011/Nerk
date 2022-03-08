from email.mime import image
from turtle import forward
from networks.utils import construct_unet
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchvision
from collections import OrderedDict

class StyleEncoder(nn.Module):
  def __init__(self, style_encoder_params):
    super().__init__()
    self.style_encoder_params = style_encoder_params
    self.style_encoder = construct_unet(self.style_encoder_params)

  def forward(self, input):
    for _, layer in enumerate(self.style_encoder):
      _, input = layer(input)

    return input


class DiscriminatorModule(nn.Module):
  def __init__(self, discriminator_params):
    super().__init__()
    self.discriminator_params = discriminator_params
    self.discriminator = construct_unet(self.discriminator_params)
    self.classifier = nn.Sequential(
      nn.Linear(discriminator_params['encoder_params'][-1], 1),
      nn.Sigmoid(),
    )
  
  def forward(self, input):
    for _, layer in enumerate(self.discriminator):
      _, input = layer(input)

    validity = self.classifier(input)

    return validity

class TextureTransfer(nn.Module):
  def __init__(self, attn_heads, in_dim=256) -> None:
    super().__init__()
    self.attention = nn.MultiheadAttention(embed_dim=in_dim, num_heads=attn_heads, batch_first=True)

  def forward(self, sketch, exemplar):
    attn_out, _ = self.attention(sketch, exemplar, exemplar) # B * 32 * 256
    texture_transfer = torch.cat((sketch, attn_out), dim=2)

    return texture_transfer


class SketchColorizer(nn.Module):
  def __init__(self, colorizer_params, style_params):
    super().__init__()
    self.colorizer_params = colorizer_params
    self.colorizer_encoder, self.colorizer_decoder = construct_unet(colorizer_params)
    self.style_encoder = construct_unet(style_params)
    self.texture_transfer = TextureTransfer(8) # TODO: parametrize

    self.skip = []

  def forward(self, sketch, exemplars):
    for i, layer in enumerate(self.colorizer_encoder):
      s, sketch = layer(sketch)
      self.skip[i] = s

    for i, layer in enumerate(self.style_encoder):
      _, exemplars = layer(exemplars)

    # Compute attention to transfer texture from exemplar to sketch
    transfer = self.texture_transfer(sketch, exemplars)

    for i, layer in enumerate(self.colorizer_decoder):
      sketch = self.colorizer_decoder(transfer, self.skip[-i])

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
  colorizer_lr: float
  discriminator_lr: float
  b1: float
  b2: float
  weight_decay: float
  '''
  def __init__(self, **hparams):
    super(SketchColoringModule, self).__init__()

    # save network hyperparameters to checkpoints
    self.save_hyperparameters()

    # initialize generator and dsicriminator
    self.Generator = SketchColorizer(self.hparams.colorizer_params, self.hparams.style_params)

    if self.hparams.train_gan:
      self.Discriminator = DiscriminatorModule(self.hparams.discriminator_params)

    # Initialiaze colored sketch generator
    self.reconstruction_loss = nn.SmoothL1Loss()
    self.adversarial_loss = nn.BCELoss()
    
    # initialize a variable to hold generated images
    self.generated_imgs = None

    # Maybe some loss with high level VGG layers
    # self.content_loss

  def forward(self, sketches, exemplars):
    colored = self.Generator(sketches, exemplars)
    return colored

  def training_step(self, batch, batch_idx, optimizer_idx=0):
    images, sketches, exemplars = self._exemplars_from_batch(batch)

    # train generator
    if optimizer_idx == 0:
      # generate images
      self.generated_imgs = self(sketches, exemplars)

      # log sampled images
      sample_imgs = self.generated_imgs[:6]
      grid = torchvision.utils.make_grid(sample_imgs)
      self.logger.experiment.add_image("generated_images", grid, 0)

      g_loss = 0
      if self.hparams.train_gan:
        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(sketches.size(0), 1)
        valid = valid.type_as(sketches)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.Discriminator(self.generated_imgs), valid)

      rec_loss = self.reconstruction_loss(self.generated_imgs, images)

      tqdm_dict = {"g_loss": g_loss, 'rec_loss': rec_loss}
      total_loss = self.hparams.g * g_loss + self.hparams.rec * rec_loss
      output = OrderedDict({"loss": total_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})

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

      fake_loss = self.adversarial_loss(self.Discriminator(self(sketches).detach()), fake)

      # discriminator loss is the average of these
      d_loss = (real_loss + fake_loss) / 2
      tqdm_dict = {"d_loss": d_loss}
      output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
      return output

  def validation_step(self, batch, batch_idx):
    sketches, images = batch
    generated_images = self(sketches)

    valid = torch.ones(sketches.size(0), 1)
    valid = valid.type_as(sketches)

    g_loss = 0
    if self.hparams.train_gan:
      g_loss = self.adversarial_loss(self.Discriminator(generated_images), valid)

    rec_loss = self.reconstruction_loss(generated_images, images)

    total_loss = self.hparams.g * g_loss + self.hparams.rec * rec_loss
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

    if self.hparams.train_gain:
      discriminator_lr = self.hparams.discriminator_lr
      opt_d = torch.optim.AdamW(self.Discriminator.parameters(), lr=discriminator_lr, betas=(b1, b2), weight_decay=self.hparams.weight_decay)

      return [opt_g, opt_d], []

    return [opt_g], []
  
  # An exemplar is an image from the same batch as the training sample
  def _exemplars_from_batch(self, batch):
    images, sketches = batch
    images = torch.repeat_interleave(images, self.hparams.num_exemplars, dim=0)
    sketches = torch.repeat_interleave(sketches, self.hparams.num_exemplars, dim=0)
    perm = torch.randperm(images.shape[0])
    exemplars = images[perm,]

    return (images, sketches, exemplars)

  def _exemplars_from_transformations(self, images, transform):
    exemplars = transform(images)
    return exemplars
