import torch
from torch import nn
import numpy as np

class ConvBlock(nn.Module):
  def __init__(self, in_c, out_c, affine=False, normalize=True, p=0):
    super().__init__()
    self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
    self.norm_1 = nn.InstanceNorm2d(out_c, affine=affine)
    self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
    self.norm_2 = nn.InstanceNorm2d(out_c, affine=affine)
    self.relu = nn.LeakyReLU(0.2, inplace=True)
    self.dropout = nn.Dropout(p=p)
    self.normalize = normalize

  def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.norm_1(x) if self.normalize else x
    x = self.relu(x)
    x = self.dropout(x)
    x = self.conv2(x)
    x = self.norm_2(x) if self.normalize else x
    x = self.relu(x)

    return x

class UNetEncoderBlock(nn.Module):
  def __init__(self, in_c, out_c, affine=False, normalize=True, p=0):
    super().__init__()
    self.conv = ConvBlock(in_c, out_c, affine=affine, normalize=normalize, p=p)
    self.pool = nn.AvgPool2d((2, 2))

  def forward(self, inputs):
    x = self.conv(inputs)
    p = self.pool(x)

    return x, p

class VisualAttention(nn.Module):
  """ Self attention Layer"""
  def __init__(self, in_dim, attn_dim):
    super(VisualAttention, self).__init__()
    self.chanel_in = in_dim
    self.attn_dim = attn_dim
    self.scale = 1.0 / np.sqrt(in_dim // self.attn_dim)
    
    self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // self.attn_dim, kernel_size=1)
    self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // self.attn_dim, kernel_size=1)
    self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size=1)
    self.gamma = nn.Parameter(torch.tensor(1, dtype=torch.float))

    self.softmax = nn.Softmax(dim=-1)

  def forward(self, sketches, exemplars):
    """
    inputs :
      x : input feature maps( B X C X W X H)
    returns :
      out : self attention value + input feature 
      attention: B X N X N (N is Width*Height)
    """
    m_batchsize, C, width, height = sketches.size()
    proj_query = self.query_conv(sketches).view(m_batchsize, -1, width * height).permute(0, 2, 1) # B X CX(N)
    proj_key = self.key_conv(exemplars).view(m_batchsize, -1, width * height) # B X C x (*W*H)
    energy = torch.bmm(proj_query, proj_key) * self.scale # transpose check
    attention = self.softmax(energy) # BX (N) X (N) 
    proj_value = self.value_conv(exemplars).view(m_batchsize, -1, width * height) # B X C X N

    out = torch.bmm(proj_value, attention.permute(0, 2, 1))
    out = out.view(m_batchsize, C, width, height)

    texture = out.clone()
    
    out = out + sketches
    return out, texture, attention

class UNetDecoderBlock(nn.Module):
  def __init__(self, in_c, out_c, affine=False, normalize=True, p=0):
    super().__init__()
    self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
    self.interpolate = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    self.conv = ConvBlock(2 * out_c, out_c, affine, normalize, p)
    self.conv_1 = nn.Conv2d(in_c, out_c, kernel_size=1)
    self.norm = nn.InstanceNorm2d(out_c, affine=affine)
    self.normalize = normalize

  def forward(self, inputs, skip, texture=None):
    x = self.up(inputs)

    if texture is not None:
      texture = self.interpolate(texture)
      texture = self.conv_1(texture)
      x = x + texture
      x = self.norm(x) if self.normalize else x

    x = torch.cat([x, skip], axis=1)
    x = self.conv(x)
    return x, texture
