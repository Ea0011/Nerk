import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms

class PerceptualLossVgg(nn.Module):
  def __init__(self,
              device=torch.device('cpu'),
              layer=35) -> None:
    super().__init__()
    self.vgg = models.vgg19(pretrained=True).features.to(device).eval()
    self.layer_out = {}
    self.fhooks = []
    self.layer = layer
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    for i, _ in enumerate(self.vgg):
      if i == self.layer:
        self.fhooks.append(self.vgg[i].register_forward_hook(self._extract_feature_forward_hook(i)))

    # truncate model to the used layer
    self.vgg = self.vgg[0:layer+1]

  def forward(self, input, target):
    input, target = self.normalize(input), self.normalize(target)
    self.vgg(input)
    input_feat_map = self.layer_out[self.layer]
    self.vgg(target)
    target_feat_map = self.layer_out[self.layer]

    gramm_matrix_input = self.gram_matrix(input_feat_map)
    gramm_matrix_target = self.gram_matrix(target_feat_map)

    return F.mse_loss(gramm_matrix_input, gramm_matrix_target)


  def _extract_feature_forward_hook(self, layer_name):
    def hook(module, input, output):
      self.layer_out[layer_name] = output

    return hook

  @staticmethod
  def gram_matrix(input):
    # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a feature map (N=c*d)
    a, b, c, d = input.size()  
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
