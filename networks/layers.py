import torch
from torch import nn

class ConvBlock(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
    self.norm_1 = nn.InstanceNorm2d(out_c, affine=True)
    self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
    self.norm_2 = nn.InstanceNorm2d(out_c, affine=True)
    self.relu = nn.LeakyReLU()

  def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.norm_1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.norm_2(x)
    x = self.relu(x)

    return x

class UNetEncoderBlock(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.conv = ConvBlock(in_c, out_c)
    self.pool = nn.MaxPool2d((2, 2))

  def forward(self, inputs):
    x = self.conv(inputs)
    p = self.pool(x)

    return x, p

class UNetDecoderBlock(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
    self.conv = ConvBlock(2 * out_c, out_c)

  def forward(self, inputs, skip):
    x = self.up(inputs)
    x = torch.cat([x, skip], axis=1)
    x = self.conv(x)
    return x

class UNet(nn.Module):
  def __init__(self):
    super().__init__()
    """ Encoder """
    self.e1 = UNetEncoderBlock(3, 32)
    self.e2 = UNetEncoderBlock(32, 64)
    self.e3 = UNetEncoderBlock(64, 128)
    self.e4 = UNetEncoderBlock(128, 256)        
    """ Bottleneck """
    self.b = ConvBlock(256, 256)         
    """ Decoder """
    self.d1 = UNetDecoderBlock(512, 256)
    self.d2 = UNetDecoderBlock(256, 128)
    self.d3 = UNetDecoderBlock(128, 64)
    self.d4 = UNetDecoderBlock(64, 32)         
    """ Classifier """
    self.outputs = nn.Conv2d(32, 3, kernel_size=1, padding=0)     

  def forward(self, inputs):
    """ Encoder """
    s1, p1 = self.e1(inputs)
    s2, p2 = self.e2(p1)
    s3, p3 = self.e3(p2)
    s4, p4 = self.e4(p3)
    """ Bottleneck """
    b = self.b(p4)         
    """ Decoder """
    d1 = self.d1(b, s4)
    d2 = self.d2(d1, s3)
    d3 = self.d3(d2, s2)
    d4 = self.d4(d3, s1)         
    """ Painter """
    outputs = self.outputs(d4)        
    return outputs