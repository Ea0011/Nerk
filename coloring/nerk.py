from PIL import Image
from processing.util import expand2square
from processing.transforms import InputTransform, RandomSketch, Sketch
from networks.modules import SketchColorizer
import json
import torch
from torchsummary import summary
import torchvision.transforms as T
import torchvision.transforms.functional as F
from skimage import morphology
import matplotlib.pyplot as plt

class Nerk():
  InterpolationMode = T.InterpolationMode

  def __init__(self, model_path=None, params_path="./params.json"):
    assert model_path is not None and params_path is not None
    colorizer_params, style_params = None, None

    with open(params_path, 'r') as params_json:
      params = json.load(params_json)
      colorizer_params, style_params = params["colorizer_params"], params["style_params"]

    self.model = SketchColorizer(colorizer_params, style_params)
    self.load_model_ckpt(model_path)

  def load_model_ckpt(self, ckpt_path=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    self.model.load_state_dict(ckpt)
    self.model.eval()

  def load_image(self, image_path=None, expand_to_square=True):
    img = Image.open(image_path)
    if expand_to_square:
      img = expand2square(img)

    return F.convert_image_dtype(F.pil_to_tensor(img))[:3, :, :]

  def prepare_input(self, image, size=(256, 256)):
    img = InputTransform(size)(image)
    return img

  def sketch(self, image, sketch_params=[0.98, 200, -0.1, 1.6, 0.8, True], remove_artifacts=True, strength=32):
    sketch = Sketch(params=sketch_params)(image[:3, :, :])

    if remove_artifacts:
      sketch = morphology.remove_small_holes(sketch.to(bool).numpy(), strength, connectivity=6)
      sketch = F.to_tensor(sketch[0]).float()
    return sketch

  def randomized_sketch(self, image, hatch=False):
    sketch = RandomSketch(hatch_enabled=hatch, hatch_dir="./processing/textures/")(image)
    return sketch

  def paint(self, sketch, exemplar):
    with torch.no_grad():
      painting = self.model(sketch.unsqueeze(0), exemplar.unsqueeze(0))[0]
      return painting.squeeze(0)

  def resize(self, image, new_size, interpolation_mode=InterpolationMode.BILINEAR):
    img = F.resize(image, new_size, interpolation_mode)
    return img

  def plot_attention_maps(self, sketch, exemplar, h, w):
    with torch.no_grad():
      attn_maps = model(sketch.unsqueeze(0), exemplar.unsqueeze(0))[-1].squeeze(0)
      upsample = torch.nn.Upsample(size=512, mode='nearest')
      attn_map = torch.zeros(1, 1, 512, 512)
      for x in range(h-25, h+25):
        for y in range(w-25, w+25):
          original_size_map = upsample(attn_maps[x + y // 32].view((1, 1, 32, 32)))
          attn_map = attn_map + torch.abs(original_size_map)

      plt.imshow(attn_map[0][0], cmap='hot', interpolation='nearest')
      plt.show()

  def demonstrate_painting(self, sketch, exemplar, painting):
    f, axarr = plt.subplots(1, 3, figsize=(32,32))
    for ax in axarr:
      ax.set_xticks([])
      ax.set_yticks([])

    axarr[0].imshow(sketch.permute(1, 2, 0), cmap="gray")
    axarr[1].imshow(exemplar.permute(1, 2, 0))
    axarr[2].imshow(painting.permute(1, 2, 0))

    plt.tight_layout()
    plt.show()

  def summary(self):
    summary(self.model, [(1, 256, 256), (3, 256, 256)])

  


if __name__ == "__main__":
  size = (512, 512)
  nerk = Nerk(model_path="./models/colorizer-stable.pth", params_path="./coloring/params.json")
  img = nerk.resize(nerk.load_image("./sketches/sample4.jpg"), size)
  exemplar = nerk.resize(nerk.load_image("./portraits/18362970.jpg"), size)
  sketch = nerk.sketch(img, sketch_params=[0.98, 200, -0.1, 1.6, 0.8, True], remove_artifacts=True, strength=32)
  sketch = nerk.resize(sketch, size, Nerk.InterpolationMode.BILINEAR)
  painting = nerk.paint(sketch, exemplar)

  nerk.demonstrate_painting(sketch, exemplar, painting)

