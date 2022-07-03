import PIL
import numpy as np
from skimage.filters import threshold_yen
import torchvision.transforms as T
import torch

hatch_transform = T.Compose([
  T.PILToTensor(),
  T.ConvertImageDtype(torch.float),
])

def xdog(im, gamma=0.98, phi=200, eps=-0.1, k=1.6, sigma=0.8, binarize=True):
  # Source : https://github.com/CemalUnal/XDoG-Filter
  # Reference : XDoG: An eXtended difference-of-Gaussians compendium including advanced image stylization
  # Link : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.365.151&rep=rep1&type=pdf
  if im.shape[0] == 3:
    im = T.Grayscale()(im)
  imf1 = T.GaussianBlur(kernel_size=7, sigma=sigma)(im)
  imf2 = T.GaussianBlur(kernel_size=7, sigma=sigma * k)(im)
  imdiff = imf1 - gamma * imf2
  imdiff = (imdiff < eps) * 1.0  + (imdiff >= eps) * (1.0 + np.tanh(phi * imdiff))
  imdiff -= imdiff.min()
  imdiff /= imdiff.max()
  if binarize:
    t = threshold_yen(imdiff.squeeze(0).numpy())
    imdiff = torch.where(imdiff > t, 1.0, 0.0)

  return imdiff

def hatch(image, texture_path):
  """
  A naive hatching implementation that takes an image and returns the image in 
  the style of a drawing created using hatching.
  image: an n x m single channel matrix.
  
  returns: an n x m single channel matrix representing a hatching style image.
  """
  params = [0.99, 400, -0.1, 1.6, 0.8, False]
  xdogImage = xdog(image, *params)

  hatchTexture = PIL.Image.open(texture_path)
  hatchTexture = hatch_transform(hatchTexture)

  _, height, width = xdogImage.shape

  croppedTexture = hatchTexture[:, 0:height, 0:width]

  return 0.7 * xdogImage + 0.3 * croppedTexture 

'''
increasing sigma increases the strength of lines
'''
PARAM_DEFAULT = [0.98, 200, -0.1, 1.6, 0.8, True]
