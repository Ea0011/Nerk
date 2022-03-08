from processing.XDoG import xdog, PARAM_DEFAULT
import numpy as np
from torchvision import transforms, utils
import torch

class Sketch():
  def __init__(self, params = PARAM_DEFAULT):
    self.params = params
  
  def __call__(self, image):
    sketch = xdog(image, *self.params)

    return sketch

r'''
An XDog filter with random Gaussian standard deviation
'''
class RandomSketch():
  def __init__(self, params = PARAM_DEFAULT):
    self.params = params
    self.params[-2] = np.random.uniform(0.5, 2.5)
  
  def __call__(self, image):
    sketch = xdog(image, *self.params)

    return sketch

class InputTransform():
  def __init__(self, size) -> None:
    self.transform = transforms.Compose([
      transforms.Resize(size),
      transforms.PILToTensor(),
      transforms.ConvertImageDtype(torch.float)
    ])
  
  def __call__(self, image):
    return self.transform(image)
    