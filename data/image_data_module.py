from data.image_folder_dataset import ImageFolderDataset
from processing.XDoG import PARAM_DEFAULT
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

class ImageDataModule(pl.LightningDataModule):
  def __init__(self,
              data_dir = "portraits/",
              batch_size = 32,
              split_ratios = [.8, .1, .1],
              image_size=128,
              training_sizes=None,
              hatch_pattern_path="./processing/textures/",
              sketch_params=PARAM_DEFAULT,
              *args,
              **kwargs):

    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.split_ratios = split_ratios
    self.image_size = image_size
    self.training_sizes = training_sizes
    self.hatch_pattern_path = hatch_pattern_path

  def prepare_data(self):
    self.train_set, self.val_set, self.test_set = \
      ImageFolderDataset(
        path=self.data_dir,
        mode='train',
        ratios=self.split_ratios,
        image_size=self.image_size,
        training_sizes=self.training_sizes,
        hatch_pattern_path=self.hatch_pattern_path), \
      ImageFolderDataset(
        path=self.data_dir,
        mode='val',
        ratios=self.split_ratios,
        image_size=self.image_size,
        hatch_pattern_path=self.hatch_pattern_path), \
      ImageFolderDataset(
        path=self.data_dir,
        mode='test',
        ratios=self.split_ratios,
        image_size=self.image_size,
        hatch_pattern_path=self.hatch_pattern_path)

  def setup(self, stage = None):
    self.prepare_data()

  def train_dataloader(self):
    return DataLoader(
      self.train_set,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=2,)

  def val_dataloader(self):
    return DataLoader(
      self.val_set,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=2,)

  def test_dataloader(self):
    return DataLoader(
      self.test_set,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=2,)

  def teardown(self, stage = None):
    pass
