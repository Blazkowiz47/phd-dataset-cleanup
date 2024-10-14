# example of calculating the frechet inception distance in Keras
import tensorflow as tf
import os
import keras
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
import PIL

        

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)
 
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

# prepare the inception v3 model
tf.reset_default_graph()
keras.backend.clear_session()

config = tf.ConfigProto()
config.allow_soft_placement=True 
config.gpu_options.per_process_gpu_memory_fraction=0.8
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
ref_imgs = []
rec_imgs = []
g = os.walk('G:/Reconstructions')
for path,d,filelist in g:  
	cnt=0
	for filename in filelist:
		ref_imgs.append(np.array(PIL.Image.open('G:/aligned_images/'+filename).convert('RGB')).astype('float32'))
		rec_imgs.append(np.array(PIL.Image.open('G:/Reconstructions/'+filename).convert('RGB')).astype('float32'))
		cnt=cnt+1
		print('%d/%d' % (cnt, len(filelist)), end='\r')

# for l in range(len(fns_ref)):
#     ref_imgs.append(np.array(PIL.Image.open('aligned_images/'+fns_ref[l]).convert('RGB')).astype('float32'))
#     rec_imgs.append(np.array(PIL.Image.open('generated_images/'+fns_ref[l]).convert('RGB')).astype('float32'))
#     print('%d/%d' % (l, len(fns_ref)), end='\r')

# resize images
ref_imgs = scale_images(ref_imgs, (299,299,3))
rec_imgs = scale_images(rec_imgs, (299,299,3))

# print('Scaled', images1.shape, images2.shape)
# pre-process images
ref_imgs = preprocess_input(ref_imgs)
rec_imgs = preprocess_input(rec_imgs)

# fid between images1 and images1
fid = calculate_fid(model, ref_imgs, rec_imgs)
print('FID %.3f' % fid)