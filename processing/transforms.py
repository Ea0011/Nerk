from processing.XDoG import hatch, xdog, PARAM_DEFAULT
import numpy as np
from torchvision import transforms
import torch
from kornia.color import LabToRgb
import os
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
  def __init__(self, params=PARAM_DEFAULT, hatch_dir="./textures", hatch_enabled=True):
    self.params = params
    self.hatch_patterns = []
    self.hatch_enabled = hatch_enabled

    for root, _, fnames in sorted(os.walk(hatch_dir)):
      for fname in sorted(fnames):
        path = os.path.join(root, fname)
        ext = os.path.splitext(path)[-1].lower()

        if ext in ['.png', '.jpg', '.jpeg']:
          path = os.path.join(root, fname)
          self.hatch_patterns.append(path)
  
  def __call__(self, image):
    do_hatch = np.random.choice([True, False], 1, p=[0.5, 0.5])[0]
    
    if do_hatch and self.hatch_enabled:
      texture_path = np.random.choice(self.hatch_patterns, 1)[0]
      sketch = hatch(image, texture_path=texture_path)

      return sketch

    self.params[0] = 0.98
    self.params[-2] = np.random.uniform(0.8, 1.2)
    self.params[1] = np.random.choice([400, 800], 1)[0]
    self.params[-3] = np.random.uniform(1.4, 1.6)
    self.params[-1] = np.random.choice([True, False], 1, p=[0.7, 0.3])[0]
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