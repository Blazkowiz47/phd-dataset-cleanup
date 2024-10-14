import numpy as np
import PIL
import os

def generate_composited_full_image(latent_vector, mask_dir):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    mask = PIL.Image.open(mask_dir)
    mask.filter(ImageFilter.GaussianBlur(5))
    mask = np.array(mask)/255
    img_array = mask*np.array(img_array) + (1.0-mask)*np.array(orig_img)
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img


