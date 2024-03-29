from networks.layers import VisualAttention
from networks.losses import PerceptualLossVgg, TextureConsistencyLoss
from networks.utils import construct_unet
from processing.transforms import DenormalizeLABImage, DenormalizeRGBImage, OutputTransform
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchvision
from collections import OrderedDict
from kornia.geometry.transform import get_tps_transform, warp_image_tps
from kornia.losses import TotalVariation, SSIMLoss

class DiscriminatorModule(nn.Module):
  def __init__(self, discriminator_params):
    super().__init__()
    self.discriminator_params = discriminator_params
    self.discriminator, self.discriminator_bottleneck = construct_unet(self.discriminator_params)
    out_dim = 2 * self.discriminator_params['encoder_blocks'][-1]['out_c']
    self.classifier = nn.Sequential(
      nn.Conv2d(out_dim, 1, kernel_size=1, padding=0),
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
    self.normalization = nn.BatchNorm2d(in_dim)

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
    self.texture_transfer = TextureTransfer(4, 2 * self.colorizer_params['encoder_blocks'][-1]['out_c'])

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

    sketch_for_loss = sketch.clone()

    # Compute attention to transfer texture from exemplar to sketch
    sketch, texture, attn_out_weights = self.texture_transfer(sketch, exemplars)
    texture_for_loss = texture.clone()

    for i, layer in enumerate(self.colorizer_decoder):
      sketch, texture = layer(sketch, skip[-(i + 1)], texture)

    for i, layer in enumerate(self.colorizer_output):
      sketch = layer(sketch)

    return sketch, texture_for_loss, sketch_for_loss, attn_out_weights

class PaintCorrectiveModule(nn.Module):
  def __init__(self, corrective_params):
    super().__init__()
    self.corrective_params = corrective_params
    self.corrective_encoder, self.corrective_bottleneck, self.corrective_decoder, self.corrective_output = construct_unet(self.corrective_params)
  
  def forward(self, input):
    skip = []

    for i, layer in enumerate(self.corrective_encoder):
      s, input = layer(input)
      skip.append(s)

    for i, layer in enumerate(self.corrective_bottleneck):
      input = layer(input)

    for i, layer in enumerate(self.corrective_decoder):
      input, _ = layer(input, skip[-(i + 1)])

    for i, layer in enumerate(self.corrective_output):
      input = layer(input)

    return input

class PaintCorrectionModule(pl.LightningModule):
  def __init__(self, device, state_dict_path=None, **hparams):
    super(PaintCorrectionModule, self).__init__()
    # save network hyperparameters to checkpoints
    self.save_hyperparameters()
    # initialize generator and dsicriminator
    self.Generator = SketchColorizer(self.hparams.colorizer_params, self.hparams.style_params)
    self.Corrective = PaintCorrectiveModule(self.hparams.corrective_params)

    # init params
    self.Corrective.apply(PaintCorrectionModule.weight_init)
    if state_dict_path is not None:
      self.restore_generator_params(state_dict_path)

    # Freeze pre trained generator network
    for p in self.Generator.parameters():
      p.requires_grad = False

    # Initialiaze colored sketch generator
    self.color_loss = nn.L1Loss()
    self.perceptual_loss = PerceptualLossVgg(device=device, layer=self.hparams.perceptual_layer)

    self.lab_to_rgb = OutputTransform()
    self.denormalize_lab = DenormalizeLABImage()
    self.denormalize_rgb = DenormalizeRGBImage()
    
    # initialize a variable to hold generated images
    self.generated_imgs = None

  def restore_generator_params(self, state_dict_path):
    state = torch.load(state_dict_path)
    self.Generator.load_state_dict(state)

  def weight_init(m):
    if isinstance(m, nn.Conv2d):
      torch.nn.init.xavier_uniform_(m.weight)
      torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.ConvTranspose2d):
      torch.nn.init.xavier_uniform_(m.weight)
      torch.nn.init.zeros_(m.bias)

  def forward(self, sketches, exemplars):
    colored, *rest = self.Generator(sketches, exemplars)
    colored = self.lab_to_rgb(colored)
    colored = self.Corrective(colored)
    return colored

  def training_step(self, batch, batch_idx):
    images, sketches, exemplars, images_lab = self._exemplars_from_transformations(batch) if \
      self.hparams.exemplar_method == "self" else self._exemplars_from_batch(batch)

    # generate images
    self.generated_imgs, *rest = \
      self.Generator(sketches.clone(), exemplars.clone())
    self.generated_imgs = self.lab_to_rgb(self.generated_imgs)
    self.generated_imgs = self.Corrective(self.generated_imgs)

    # log sampled images
    if self.global_step % 32 == 0:
      sample_imgs = self.generated_imgs[:6].detach().clone()
      imgs_to_plot = torch.cat([sample_imgs], dim=0)
      grid = torchvision.utils.make_grid(imgs_to_plot)
      self.logger.experiment.add_image("generated_images", grid, self.global_step)

    perc_loss = self.perceptual_loss(self.generated_imgs.clone(), images.clone()) if self.hparams.perc > 0 else 0
    color_loss = self.color_loss(self.generated_imgs, images)
    total_loss = color_loss + self.hparams.perc * perc_loss

    log = {
      "color_loss": color_loss,
      "perceptual_loss": perc_loss,
      "loss": total_loss,
    } 

    output = OrderedDict({
      "color_loss": color_loss,
      "perceptual_loss": perc_loss,
      "loss": total_loss,
    })
    self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log_dict(log, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    return output

  def validation_step(self, batch, batch_idx):
    images, sketches, exemplars, images_lab = self._exemplars_from_transformations(batch) if \
      self.hparams.exemplar_method == "self" else self._exemplars_from_batch(batch)

    generated_images, *rest = self.Generator(sketches.clone(), exemplars.clone())
    generated_images = self.lab_to_rgb(generated_images)
    generated_images = self.Corrective(generated_images)

    color_loss = self.color_loss(generated_images, images)

    self.log('val_loss', color_loss, prog_bar=True)

    return color_loss

  def configure_optimizers(self):
    lr = self.hparams.lr
    b1 = self.hparams.b1
    b2 = self.hparams.b2
    opt = torch.optim.AdamW(
      self.Corrective.parameters(),
      lr=lr,
      betas=(b1, b2),
      weight_decay=self.hparams.weight_decay,)

    scheduler = {
      'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt,
        T_0=1000,
        verbose=False,
        eta_min=self.hparams.min_lr),
      'interval': 'step',
    }

    return [opt], [scheduler]
  
  # An exemplar is an image from the same batch as the training sample
  def _exemplars_from_batch(self, batch):
    images, sketches, images_lab = batch
    images = torch.repeat_interleave(images, self.hparams.num_exemplars, dim=0)
    images_lab = torch.repeat_interleave(images_lab, self.hparams.num_exemplars, dim=0)
    sketches = torch.repeat_interleave(sketches, self.hparams.num_exemplars, dim=0)
    perm = torch.randperm(images.shape[0])
    exemplars = images_lab[perm,].clone() # retain only color information for exemplars

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
    points_dst = points_src + torch.rand(images_lab.shape[0], 5, 2).type_as(images_lab) * (-0.8) + 0.4
    # note that we are getting the reverse transform: dst -> src
    kernel_weights, affine_weights = get_tps_transform(points_dst, points_src)
    exemplars = warp_image_tps(images_lab.clone(), points_src, kernel_weights, affine_weights)

    return (images, sketches, exemplars, images_lab)

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
      torch.zeros((1, 1, 64, 64), dtype=torch.float), # sktech
      torch.zeros((1, 3, 64, 64), dtype=torch.float), # exemplar
    ]

    # initialize generator and dsicriminator
    self.Generator = SketchColorizer(self.hparams.colorizer_params, self.hparams.style_params)
    self.Discriminator = DiscriminatorModule(self.hparams.discriminator_params)

    # Initialiaze colored sketch generator
    self.color_loss = nn.L1Loss()
    self.reconstruction_loss = nn.L1Loss()
    self.adversarial_loss = nn.BCEWithLogitsLoss() if self.hparams.gan_loss == 'BCE' else nn.MSELoss()
    self.discirminator_loss = nn.BCEWithLogitsLoss() if self.hparams.gan_loss == 'BCE' else nn.MSELoss()
    self.perceptual_loss = PerceptualLossVgg(device=device, layer=self.hparams.perceptual_layer)
    self.texture_loss = TextureConsistencyLoss(
      device=device, 
      texture_layer=self.Generator.texture_transfer,
      style_encoder=self.Generator.style_encoder,
      style_bottleneck=self.Generator.style_bottleneck)
    self.total_variation_loss = TotalVariation()
    self.struct_similarity_loss = SSIMLoss(window_size=5, max_val=1.0) # SSIM index for validation

    self.lab_to_rgb = OutputTransform()
    self.denormalize_lab = DenormalizeLABImage()
    self.denormalize_rgb = DenormalizeRGBImage()
    
    # initialize a variable to hold generated images
    self.generated_imgs = None
    self.texture_for_loss = None
    self.sketch_for_loss = None

    # Maybe some loss with high level VGG layers
    # self.content_loss

  def forward(self, sketches, exemplars):
    colored, texture_for_loss, sketch_for_loss, attn = self.Generator(sketches, exemplars)
    return colored, texture_for_loss, sketch_for_loss, attn

  def training_step(self, batch, batch_idx, optimizer_idx):
    images, sketches, exemplars, images_lab = self._exemplars_from_transformations(batch) if \
      self.hparams.exemplar_method == "self" else self._exemplars_from_batch(batch)

    # train generator
    if optimizer_idx == 0:
      # generate images
      self.generated_imgs, self.texture_for_loss, self.sketch_for_loss, *rest = \
        self(sketches.clone(), exemplars.clone())

      # log sampled images
      if self.global_step % 32 == 0:
        sample_imgs = self.generated_imgs[:6].detach().clone()
        sample_imgs = torch.clamp(sample_imgs, 0, 1)
        exemplar_imgs = exemplars[:6].clone()
        imgs_to_plot = torch.cat([sample_imgs, exemplar_imgs], dim=0)
        grid = torchvision.utils.make_grid(imgs_to_plot)
        self.logger.experiment.add_image("generated_images", grid, self.global_step)

      color_loss = self.color_loss(self.generated_imgs.clone(), images.clone())

      perc_loss = self.perceptual_loss(self.generated_imgs.clone(), images) if self.hparams.perc > 0 else 0
      texture_loss = self.texture_loss( # May need to detach some parts
        self.texture_for_loss,
        self.sketch_for_loss,
        self.generated_imgs.clone())
      total_variation_loss = self.total_variation_loss(self.generated_imgs.clone())
      total_variation_loss = total_variation_loss.mean()

      g_loss = 0
      if self.hparams.train_gan:
        # adversarial loss is binary cross-entropy
        cgan_input = torch.cat([sketches.clone(), self.generated_imgs.clone()], axis=1)
        cgan_out = self.Discriminator(cgan_input)
        valid = torch.ones_like(cgan_out)
        valid = valid.type_as(sketches)

        g_loss = self.adversarial_loss(cgan_out, valid)

      total_loss = self.hparams.g * g_loss + self.hparams.perc * perc_loss + \
        self.hparams.color * color_loss + self.hparams.texture * texture_loss + self.hparams.variation * total_variation_loss

      generator_log = {
        "g_loss": g_loss,
        'perc_loss': perc_loss,
        'total_loss': total_loss,
        'color_loss': color_loss,
        'texture_loss': texture_loss,
        'total_variation_loss': total_variation_loss,
      } 

      output = OrderedDict({
        "loss": total_loss,
        'color_loss': color_loss,
        "perceptual_loss": perc_loss,
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
      cgan_input = torch.cat([sketches.clone(), images.clone()], axis=1)
      cgan_out = self.Discriminator(cgan_input)
      valid = torch.ones_like(cgan_out)
      valid = valid.type_as(sketches)
      real_loss = self.discirminator_loss(cgan_out, valid)

      # how well can it label as fake?
      self.generated_imgs, *rest = self(sketches.clone(), exemplars.clone())

      cgan_input = torch.cat([sketches.clone(), self.generated_imgs.clone()], axis=1)
      cgan_out = self.Discriminator(cgan_input)
      fake = torch.zeros_like(cgan_out)
      fake = fake.type_as(images)
      fake_loss = self.discirminator_loss(cgan_out, fake)

      # discriminator loss is the average of these
      d_loss = (real_loss + fake_loss) / 2
      discriminator_log = {"d_loss": d_loss, 'd_loss_fake': fake_loss, 'd_loss_real': real_loss}
      output = OrderedDict({"loss": d_loss, "progress_bar": discriminator_log, "log": discriminator_log})
      self.log_dict(discriminator_log, on_step=True, on_epoch=True, prog_bar=True, logger=True)
      return output

  def validation_step(self, batch, batch_idx):
    images, sketches, images_lab = batch

    generated_images, *rest = self(sketches.clone(), images.clone())

    g_loss = 0
    if self.hparams.train_gan:
      cgan_input = torch.cat([sketches.clone(), generated_images.clone()], axis=1)
      cgan_out = self.Discriminator(cgan_input)
      valid = torch.ones_like(cgan_out)
      valid = valid.type_as(sketches)
      g_loss = self.adversarial_loss(cgan_out, valid)

    color_loss = self.color_loss(generated_images.clone(), images.clone())

    # View results of validation
    sample_imgs = generated_images[:6].detach().clone()
    sample_imgs = torch.clamp(sample_imgs, 0, 1)
    exemplar_imgs = images[:6]
    imgs_to_plot = torch.cat([sample_imgs, exemplar_imgs], dim=0)
    grid = torchvision.utils.make_grid(imgs_to_plot)
    self.logger.experiment.add_image("val_generated_images", grid, self.global_step)

    ssim_loss = self.struct_similarity_loss(generated_images.clone(), images.clone())

    perc_loss = self.perceptual_loss(generated_images, images) if self.hparams.perc > 0 else 0
    total_loss = ssim_loss + self.hparams.perc * perc_loss + \
      self.hparams.color * color_loss

    val_log = {
      "val_g_loss": g_loss,
      'val_perc_loss': perc_loss,
      'val_total_loss': total_loss,
      'val_color_loss': color_loss,
      'val_ssim_loss': ssim_loss,
    } 

    self.log('val_loss', total_loss, prog_bar=True)
    self.log_dict(val_log, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    return total_loss

  def configure_optimizers(self):
    colorizer_lr = self.hparams.colorizer_lr
    b1 = self.hparams.b1
    b2 = self.hparams.b2
    disc_b1 = self.hparams.disc_b1
    disc_b2 = self.hparams.disc_b2
    opt_g = torch.optim.AdamW(
      self.Generator.parameters(),
      lr=colorizer_lr,
      betas=(b1, b2),
      weight_decay=self.hparams.weight_decay,)

    generator_scheduler = {
      'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_g,
        T_0=1000,
        verbose=False,
        eta_min=self.hparams.min_lr),
      'interval': 'step',
    }

    if self.hparams.train_gan:
      discriminator_lr = self.hparams.discriminator_lr
      opt_d = torch.optim.AdamW(
        self.Discriminator.parameters(),
        lr=discriminator_lr,
        betas=(disc_b1, disc_b2),
        weight_decay=self.hparams.weight_decay,)

      discriminator_scheduler = {
        'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
          opt_d,
          T_0=1000,
          verbose=False,
          eta_min=self.hparams.min_lr),
        'interval': 'step',
      }

      return opt_g, opt_d

    return [opt_g], [generator_scheduler]
  
  # An exemplar is an image from the same batch as the training sample
  def _exemplars_from_batch(self, batch):
    images, sketches, images_lab = batch
    images = torch.repeat_interleave(images, self.hparams.num_exemplars, dim=0)
    images_exemp = torch.repeat_interleave(images, self.hparams.num_exemplars, dim=0)
    sketches = torch.repeat_interleave(sketches, self.hparams.num_exemplars, dim=0)
    perm = torch.randperm(images.shape[0])
    exemplars = images_exemp[perm,].clone() # retain only color information for exemplars

    return (images, sketches, exemplars, images_exemp)

  def _exemplars_from_transformations(self, batch):
    images, sketches, images_lab = batch
    images = torch.repeat_interleave(images, self.hparams.num_exemplars, dim=0)
    images_exemp = torch.repeat_interleave(images, self.hparams.num_exemplars, dim=0)
    sketches = torch.repeat_interleave(sketches, self.hparams.num_exemplars, dim=0)

    xs = torch.tensor([-1.0, 1, 0, -1, 1])
    ys = torch.tensor([-1.0, -1, 0, 1, 1])

    points_src = torch.stack((xs, ys), 1).unsqueeze(0)
    points_src = torch.repeat_interleave(points_src, images_exemp.shape[0], dim=0).type_as(images_exemp)
    points_dst = points_src + torch.rand(images_exemp.shape[0], 5, 2).type_as(images_exemp) * (-0.8) + 0.4
    # note that we are getting the reverse transform: dst -> src
    kernel_weights, affine_weights = get_tps_transform(points_dst, points_src)
    exemplars = warp_image_tps(images_exemp.clone(), points_src, kernel_weights, affine_weights)

    return (images, sketches, exemplars, images_exemp)