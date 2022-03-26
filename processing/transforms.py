from processing.XDoG import xdog, PARAM_DEFAULT
import numpy as np
from torchvision import transforms
import torch
from kornia.color import LabToRgb
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
  
  def __call__(self, image):
    self.params[-2] = np.random.uniform(0.5, 1.6)
    self.params[1] = np.random.choice([100, 200, 600, 800], 1)[0]
    sketch = xdog(image, *self.params)

    return sketch

class InputTransform():
  def __init__(self, size) -> None:
    self.transform = transforms.Compose([
      transforms.Resize(size),
      transforms.PILToTensor(),
      transforms.ConvertImageDtype(torch.float),
    ])
  
  def __call__(self, image):
    return self.transform(image)

class OutputTransform():
  def __init__(self) -> None:
    self.transform = LabToRgb()

  def __call__(self, image_lab):
    # Denormalize input to transform to back to rgb data
    image_lab[:,0] = (image_lab[:,0] + 1) * 50
    image_lab[:,1:] = image_lab[:,1:] * 127

    return self.transform(image_lab, clip=True)

class DenormalizeLABImage():
  def __call__(self, image_lab):
    image_lab[:,0] = (image_lab[:,0] + 1) * 50
    image_lab[:,1:] = image_lab[:,1:] * 127

    return image_lab

class DenormalizeRGBImage():
  def __call__(self, image_rgb):
    return image_rgb * 255