import numpy as np
import os
from PIL import Image
from processing.XDoG import PARAM_DEFAULT

from processing.transforms import InputTransform, RandomSketch, Sketch

class ImageFolderDataset():
  def __init__(self,
              path='data/portraits',
              mode='train',
              ratios=[.8, .1, .1], # train/val/test ratios for data split
              transform=None,
              sketch_params=PARAM_DEFAULT,
              *args,
              **kwargs):

    self.images = self.make_dataset(
      directory=path,
    )

    # For deterministic sampling
    r = np.random.RandomState(1234)
    r.shuffle(self.images)

    train, val, test = np.split(self.images, 
      [int(ratios[0] * len(self.images)),
       int((ratios[0] + ratios[1]) * len(self.images)),]
    )

    if mode == 'train':
      self.images = train
    elif mode == 'val':
      self.images = val
    elif mode == 'test':
      self.images = test

    # transform function that we will apply later for data preprocessing
    self.transform = transform
    self.sketch_transform = RandomSketch(sketch_params)

  @staticmethod
  def make_dataset(directory):
    """
    Create the image dataset by preparaing a list of samples
    Images are sorted in an ascending order by class and file name
    :param directory: root directory of the dataset
    :returns: (images) where:
        - images is a list containing paths to all images in the dataset, NOT the actual images
    """
    images = []

    for root, _, fnames in sorted(os.walk(directory)):
      for fname in sorted(fnames):
        path = os.path.join(root, fname)
        images.append(path)

    return images

  def __len__(self):
    length = len(self.images)
    return length

  @staticmethod
  def load_image(image_path):
    """Load image from image_path as numpy array"""
    return Image.open(image_path)

  def __getitem__(self, index):
    image = InputTransform(size=512)(self.load_image(self.images[index]))
    sketch = self.sketch_transform(image)

    # Not sure what kind of transform can be applied to target
    # image_transformed = self.transform(image)
    return image,sketch
