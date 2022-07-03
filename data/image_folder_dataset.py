import numpy as np
import os
from PIL import Image
from processing.XDoG import PARAM_DEFAULT
from kornia.color import RgbToLab
from processing.transforms import InputTransform, RandomSketch, Sketch

class ImageFolderDataset():
  def __init__(self,
              path='data/portraits',
              mode='train',
              ratios=[.8, .1, .1], # train/val/test ratios for data split
              transform=None,
              sketch_params=PARAM_DEFAULT,
              image_size=None,
              training_sizes=None,
              continuation_ratio=None,
              hatch_pattern_path="../processing/textures/",
              *args,
              **kwargs):

    self.image_size = image_size
    self.images = self.make_dataset(
      directory=path,
    )
    self.rgb_to_lab = RgbToLab()

    # For deterministic sampling
    r = np.random.RandomState(432)
    r.shuffle(self.images)

    train, val, test = np.split(self.images, 
      [int(ratios[0] * len(self.images)),
       int((ratios[0] + ratios[1]) * len(self.images)),]
    )

    if mode == 'train':
      self.images = train
      if continuation_ratio:
        # Circle the list to start with samples not used
        assert(continuation_ratio < 1)

        non_used_portion = len(self.images) * continuation_ratio
        non_used_images = self.images[non_used_portion:]
        used_images = self.images[:non_used_portion]

        self.images = non_used_images + used_images
    elif mode == 'val':
      self.images = val
    elif mode == 'test':
      self.images = test

    # transform function that we will apply later for data preprocessing
    self.transform = transform
    self.sketch_transform = RandomSketch(sketch_params, hatch_dir=hatch_pattern_path)
    self.training_sizes = training_sizes

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
    image = None
    if self.training_sizes is not None:
      size = int(np.random.choice(self.training_sizes, 1)[0])
      image = InputTransform(size=size)(self.load_image(self.images[index]))
    else:
      image = InputTransform(size=self.image_size)(self.load_image(self.images[index]))

    sketch = self.sketch_transform(image) # For condditional GAN
    img_lab = self.rgb_to_lab(image) # For chrominance loss

    # normalize input to model
    img_lab[0,] = 2*(img_lab[0,] / 100) - 1
    img_lab[1:,] = img_lab[1:,] / 127

    # Not sure what kind of transform can be applied to target
    # image_transformed = self.transform(image)
    return image,sketch,img_lab
