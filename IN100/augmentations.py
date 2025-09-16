# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""

import numpy as np
import skimage as sk
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms as trn
import sys
from skimage.filters import gaussian
from PIL import Image as PILImage
import cv2
from io import BytesIO
# ImageNet code should change this value
IMAGE_SIZE = 224
def set_image_size(size):
    global IMAGE_SIZE
    IMAGE_SIZE = size

def get_image_size():
    return IMAGE_SIZE

def get_frost_image(filename):
    """Load the frost image from file and cache it for future use."""
    if filename not in FROST_CACHE:
        FROST_CACHE[filename] = cv2.imread(filename)
    return FROST_CACHE[filename]
#Global caching for frost images ---
FROST_FILENAMES = [
    './frost1.png', './frost2.png', './frost3.png',
    './frost4.jpg', './frost5.jpg', './frost6.jpg'
]
FROST_CACHE = {}
#Global buffer for JPEG compression
GLOBAL_BUFFER = BytesIO()

convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])

def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    x = np.uint8(x)
    return convert_img(x)

def frost(x, severity=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    filename = FROST_FILENAMES[idx]
    frost = cv2.imread(filename)
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(0, frost.shape[1] - 224)
    frost = frost[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]

    x = np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)

    return convert_img(x)

def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    x = x.resize((int(224 * c), int(224 * c)), PILImage.BOX)
    x = x.resize((224, 224), PILImage.BOX)
    return x

def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.05,1,1), (0.25,1,1), (0.4,1,1), (0.25,1,2), (0.4,1,2)][severity - 1]
    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0]) * 255)
    # Locally shuffle pixels
    for i in range(c[2]):
        for h in range(224 - c[1], c[1], -1):
            for w in range(224 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # Swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    x = np.clip(gaussian(x / 255., sigma=c[0]), 0, 1) * 255
    x = np.uint8(x)
    return convert_img(x)

def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]
    # Reuse the global buffer to avoid repeated allocation:
    GLOBAL_BUFFER.seek(0)
    GLOBAL_BUFFER.truncate(0)
    x.save(GLOBAL_BUFFER, 'JPEG', quality=c)
    GLOBAL_BUFFER.flush()  #flush the buffer to ensure all data is written
    GLOBAL_BUFFER.seek(0)
    x_new = PILImage.open(GLOBAL_BUFFER)
    x_new.load()  #force loading of the image
    return x_new

# def gaussian_noise(x, severity=1):
#     c = [0.04, 0.06, .08, .09, .10][severity - 1]
#     x = np.array(x) / 255.
#     x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
#     x = np.uint8(x)
#     return convert_img(x)

# def frost(x, severity=1):
#     c = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)][severity - 1]
#     idx = np.random.randint(5)
#     filename = ['./frost1.png', './frost2.png', './frost3.png', './frost4.jpg', './frost5.jpg', './frost6.jpg'][idx]
#     frost = cv2.imread(filename)
#     frost = cv2.resize(frost, (0, 0), fx=0.2, fy=0.2)
#     # randomly crop and convert to rgb
#     x_start, y_start = np.random.randint(0, frost.shape[0] - 32), np.random.randint(0, frost.shape[1] - 32)
#     frost = frost[x_start:x_start + 32, y_start:y_start + 32][..., [2, 1, 0]]
#     x =  np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)
#     x = np.uint8(x)
#     return convert_img(x)

# def pixelate(x, severity=1):
#     c = [0.95, 0.9, 0.85, 0.75, 0.65][severity - 1]

#     x = x.resize((int(32 * c), int(32 * c)), PILImage.BOX)
#     x = x.resize((32, 32), PILImage.BOX)

#     return x
# def glass_blur(x, severity=1):
#     # sigma, max_delta, iterations
#     c = [(0.05,1,1), (0.25,1,1), (0.4,1,1), (0.25,1,2), (0.4,1,2)][severity - 1]

#     x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0]) * 255)

#     # locally shuffle pixels
#     for i in range(c[2]):
#         for h in range(32 - c[1], c[1], -1):
#             for w in range(32 - c[1], c[1], -1):
#                 dx, dy = np.random.randint(-c[1], c[1], size=(2,))
#                 h_prime, w_prime = h + dy, w + dx
#                 # swap
#                 x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
#     x =np.clip(gaussian(x / 255., sigma=c[0]), 0, 1) * 255
#     x = np.uint8(x)
#     return convert_img(x)

# def jpeg_compression(x, severity=1):
#     c = [80, 65, 58, 50, 40][severity - 1]

#     output = BytesIO()
#     x.save(output, 'JPEG', quality=c)
#     x = PILImage.open(output)
#     return x
# def frost(x, severity=1):
#     c = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)][severity - 1]
#     idx = np.random.randint(5)
#     filename = FROST_FILENAMES[idx]
#     frost_img = get_frost_image(filename)
#     # Resize frost image only once per call (using caching helps if the same filename is used repeatedly)
#     # frost_img = cv2.resize(frost_img, (0, 0), fx=0.2, fy=0.2)
#     # Randomly crop a 32x32 region and convert BGR to RGB
#     x_start = np.random.randint(0, frost_img.shape[0] - 224)
#     y_start = np.random.randint(0, frost_img.shape[1] - 224)
#     frost_crop = frost_img[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]
#     x = np.clip(c[0] * np.array(x) + c[1] * frost_crop, 0, 255)
#     x = np.uint8(x)
#     return convert_img(x)

augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness, 
    gaussian_noise, pixelate, glass_blur, jpeg_compression
]

# augmentations_all = [
#     autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
#     translate_x, translate_y, color, contrast, brightness, sharpness, 
#     gaussian_noise, pixelate, glass_blur, frost, jpeg_compression
# ]
