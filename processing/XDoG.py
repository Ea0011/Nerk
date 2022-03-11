import argparse
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
import sys
import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu
import torch
import torchvision.transforms as T

def xdog(im, gamma=0.98, phi=200, eps=-0.1, k=1.6, sigma=0.8, binarize=False):
  # Source : https://github.com/CemalUnal/XDoG-Filter
  # Reference : XDoG: An eXtended difference-of-Gaussians compendium including advanced image stylization
  # Link : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.365.151&rep=rep1&type=pdf
  if im.shape[0] == 3:
    im = T.Grayscale()(im)
  imf1 = T.GaussianBlur(kernel_size=5, sigma=sigma)(im)
  imf2 = T.GaussianBlur(kernel_size=5, sigma=sigma * k)(im)
  imdiff = imf1 - gamma * imf2
  imdiff = (imdiff < eps) * 1.0  + (imdiff >= eps) * (1.0 + np.tanh(phi * imdiff))
  imdiff -= imdiff.min()
  imdiff /= imdiff.max()
  if binarize:
    th = threshold_otsu(imdiff.permute(1, 2, 0))
    imdiff = imdiff >= th

  return imdiff

'''
increasing sigma increases the strength of lines
'''
PARAM_DEFAULT = [0.98, 200, -0.1, 1.6, 0.8, False]

