from networks.layers import UNetEncoderBlock, UNetDecoderBlock
from networks.layers import ConvBlock
from torch import nn

r'''
params: [{'in_c', 'out_c'}]
'''
def construct_unet(params):
  enc_layers = []
  dec_layers = []

  if "decoder_blocks" not in params:
    for enc_dims in params["encoder_blocks"]:
      encoder = UNetEncoderBlock(enc_dims['in_c'], enc_dims['out_c'])
      enc_layers.append(encoder)
    
    bottle_neck = ConvBlock(params["encoder_blocks"][-1]['out_c'], params["encoder_blocks"][-1]['out_c'])
    enc_layers.append(bottle_neck)

    encoder = nn.ModuleList(enc_layers)

    return encoder

  encoder_blocks, decoder_blocks = params["encoder_blocks"], params["decoder_blocks"]
  for enc_dims, dec_dims in zip(encoder_blocks, decoder_blocks):
    encoder = UNetEncoderBlock(enc_dims['in_c'], enc_dims['out_c'])
    decoder = UNetDecoderBlock(dec_dims['in_c'], dec_dims['out_c'])
    enc_layers.append(encoder)
    dec_layers.append(decoder)

  bottle_neck = ConvBlock(encoder_blocks[-1]['out_c'], encoder_blocks[-1]['out_c'])
  enc_layers.append(bottle_neck)

  output = nn.Conv2d(decoder_blocks[-1]['out_c'], 3, kernel_size=1)
  act = nn.Sigmoid()

  dec_layers.append(output)
  dec_layers.append(act)

  encoder = nn.ModuleList(enc_layers)
  decoder = nn.ModuleList(dec_layers)

  return encoder, decoder
