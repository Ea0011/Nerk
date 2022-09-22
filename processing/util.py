from PIL import Image
import numpy as np

def img_to_float(img):
  as_float = np.array(img, dtype=np.float32)
  as_float = as_float / 255.
  as_float = np.expand_dims(as_float, axis=-1)

  return as_float 

def float_to_img(arr):
  scaled = arr * 255.
  scaled = np.squeeze(scaled, axis=-1)
  scaled = scaled.astype(np.uint8)
  img = Image.fromarray(scaled, mode='L')

  return img

def load_img(path):
  input_img = Image.open(path)
  input_img = input_img.convert(mode='L')
  input_arr = img_to_float(input_img)
  return input_arr

def save_img(path, img_arr):
  output = float_to_img(img_arr)
  with path.open('wb+') as f:
    output.save(f)

def expand2square(pil_img, background_color=(0,0,0)):
  width, height = pil_img.size
  if width == height:
    return pil_img
  elif width > height:
    result = Image.new(pil_img.mode, (width, width), background_color)
    result.paste(pil_img, (0, (width - height) // 2))
    return result
  else:
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result
