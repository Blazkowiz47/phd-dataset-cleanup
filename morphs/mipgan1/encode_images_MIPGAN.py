# __author__ = "Haoyu Zhang"
# __copyright__ = "Copyright (C) 2021 Norwegian University of Science and Technology"
# __license__ = "License Agreement provided by Norwegian University of Science and Technology (NTNU)" \
#               "(MIPGAN-license-210420.pdf)"
# __version__ = "1.0"

import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
from PIL import ImageFilter
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model_MIPGAN import PerceptualModel, load_ref_image
#from tensorflow.keras.models import load_model
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input

import io
import yaml
from imageio import imread
from model import get_embd
import math
import tensorflow.contrib.slim as slim
import tensorflow as tf

import csv


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def distance(embeddings1, embeddings2, distance_type='Euclidian'):
    embeddings1=embeddings1.astype(np.float64)
    embeddings2=embeddings2.astype(np.float64)
    if distance_type=='Euclidian':
        # Euclidian distance
        embeddings1 = embeddings1/np.linalg.norm(embeddings1, axis=0, keepdims=True)
        embeddings2 = embeddings2/np.linalg.norm(embeddings2, axis=0, keepdims=True)
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),0)
    elif distance_type=='Cosine':
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=0)
        norm = np.linalg.norm(embeddings1, axis=0) * np.linalg.norm(embeddings2, axis=0)
        similarity = dot/norm
        similarity = min(1,similarity)
        dist=1-similarity
        # dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric' 
        
    return dist

def run_embds(sess, images, batch_size, image_size, train_mode, embds_ph, image_ph, train_ph_dropout, train_ph_bn):
    if train_mode >= 1:
        train = True
    else:
        train = False
    batch_num = len(images)//batch_size
    left = len(images)%batch_size
    embds = []
    for i in range(batch_num):
        image_batch = images[i*batch_size: (i+1)*batch_size]
        cur_embd = sess.run(embds_ph, feed_dict={image_ph: image_batch, train_ph_dropout: train, train_ph_bn: train})
        embds += list(cur_embd)
        print('%d/%d' % (i, batch_num), end='\r')
    if left > 0:
        image_batch = np.zeros([batch_size, image_size, image_size, 3])
        image_batch[:left, :, :, :] = images[-left:]
        cur_embd = sess.run(embds_ph, feed_dict={image_ph: image_batch, train_ph_dropout: train, train_ph_bn: train})
        embds += list(cur_embd)[:left]
    print()
    print('done!')
    return np.array(embds)

def load_image_frs(dir_path, image_size):
    print('reading %s' % dir_path)
    if os.path.isdir(dir_path):
        paths = list(os.listdir(dir_path))
    else:
        paths = [dir_path]
    images = []
    images_f = []
    for path in paths:
        img = imread(dir_path+'/'+path)
        img = np.array(PIL.Image.fromarray(img).resize((image_size,image_size)))
        # img = misc.imresize(img, [image_size, image_size])
        # img = img[s:s+image_size, s:s+image_size, :]
        img_f = np.fliplr(img)
        img = img/127.5-1.0
        img_f = img_f/127.5-1.0
        images.append(img)
        images_f.append(img_f)
    fns = [os.path.basename(p) for p in paths]
    print('done!')
    return (np.array(images), np.array(images_f), fns)

def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual losses', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('morph_list_CSV', default='morph_list.csv', help='CSV file containting the morphing pairs with their FILENAME ONLY of each contributing image. Format: /<image1> , /<image2>')
    parser.add_argument('src_dir', default='aligned_images/', help='Directory with images for morphing' )
    parser.add_argument('generated_images_dir', default='generated_images/', help='Directory for storing generated images' )
    parser.add_argument('dlatent_dir', default='latent_representations/', help='Directory for storing dlatent representations')
    parser.add_argument('--data_dir', default='data', help='Directory for storing optional models')
    parser.add_argument('--mask_dir', default='masks', help='Directory for storing optional masks')
    parser.add_argument('--loss_dir', default='loss', help='Directory for storing loss values')
    parser.add_argument('--frs_config_path', type=str, default='./models/frs/configs/config_ms1m_100.yaml', help='config path, used when mode is build')
    parser.add_argument('--frs_model_path', type=str, default='models/frs/config_ms1m_100_1006k/best-m-1006000', help='model path')
    parser.add_argument('--load_last', default='', help='Start with embeddings from directory')
    parser.add_argument('--dlatent_avg', default='', help='Use dlatent from file specified here for truncation instead of dlatent_avg from Gs')
    # parser.add_argument('--model_url', default='https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ', help='Fetch a StyleGAN model to train on from this URL') # karras2019stylegan-ffhq-1024x1024.pkl
    parser.add_argument('--model_url', default='models/StyleGAN_finetuned_ICAO.pkl', help='Fetch a finetuned StyleGAN model to train on from this URL')
    parser.add_argument('--model_res', default=1024, help='The dimension of images in the StyleGAN model', type=int)
    # parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
    parser.add_argument('--optimizer', default='adam', help='Optimization algorithm used for optimizing dlatents')

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--resnet_image_size', default=256, help='Size of images for the Resnet model', type=int)
    parser.add_argument('--lr', default=0.03, help='Learning rate for perceptual model', type=float)
    parser.add_argument('--decay_rate', default=0.95, help='Decay rate for learning rate', type=float)
    parser.add_argument('--iterations', default=150, help='Number of optimization steps for each batch', type=int)
    parser.add_argument('--decay_steps', default=6, help='Decay steps for learning rate decay (as a percent of iterations)', type=float)
    parser.add_argument('--early_stopping', default=False, help='Stop early once training stabilizes', type=str2bool, nargs='?', const=True)
    parser.add_argument('--early_stopping_threshold', default=10000000, help='Stop after the loss is lower than this threshold. Disable: set a very large threshold', type=float)
    parser.add_argument('--early_stopping_ratio', default=0, help='Stop after the decrease of the loss (loss[-1]-loss[-1]) has reached the threshold. Disable: 0 ', type=float)
    parser.add_argument('--early_stopping_patience', default=10, help='Number of iterations to wait below threshold', type=int)    
    parser.add_argument('--load_resnet', default='models/finetuned_resnet.h5', help='Model to load for ResNet approximation of dlatents')
    parser.add_argument('--use_preprocess_input', default=True, help='Call process_input() first before using feed forward net', type=str2bool, nargs='?', const=True)
    parser.add_argument('--use_best_loss', default=True, help='Output the lowest loss value found as the solution', type=str2bool, nargs='?', const=True)
    parser.add_argument('--average_best_loss', default=0.5, help='Do a running weighted average with the previous best dlatents found', type=float)
    parser.add_argument('--sharpen_input', default=True, help='Sharpen the input images', type=str2bool, nargs='?', const=True)

    # Loss function options
    parser.add_argument('--use_perceptual_loss', default=0.0002, help='Use VGG multi-layer perceptual loss; 0 to disable, > 0 to scale.', type=float)
    # parser.add_argument('--use_vgg_loss', default=0, help='Use VGG perceptual loss; 0 to disable, > 0 to scale.', type=float)
    # parser.add_argument('--use_vgg_layer', default=9, help='Pick which VGG layer to use.', type=int)
    parser.add_argument('--use_pixel_loss_logcosh', default=0, help='Use logcosh image pixel loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_pixel_loss_mse', default=0, help='Use mse image pixel loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_msssim_loss', default=1, help='Use MS-SSIM perceptual loss; 0 to disable, > 0 to scale.', type=float)
    # parser.add_argument('--use_lpips_loss', default=0, help='Use LPIPS perceptual loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_l1_penalty', default=0, help='Use L1 penalty on latents; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_discriminator_loss', default=0, help='Use trained discriminator to evaluate realism.', type=float)
    # parser.add_argument('--use_adaptive_loss', default=False, help='Use the adaptive robust loss function from Google Research for pixel and VGG feature loss.', type=str2bool, nargs='?', const=True)
    parser.add_argument('--use_arcface_loss', default=10, help='Use arcface to attack frs system', type=float)    
    parser.add_argument('--use_arcface_difference_loss', default=1, help='Constrain the same similarity to each reference image', type=float)
    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=str2bool, nargs='?', const=True)
    parser.add_argument('--tile_dlatents', default=False, help='Tile dlatents to use a single vector at each scale', type=str2bool, nargs='?', const=True)
    parser.add_argument('--clipping_threshold', default=0, help='Stochastic clipping of gradient values outside of this threshold. Disable: set threshold < 0', type=float)

    # Masking params
    # parser.add_argument('--load_mask', default=False, help='Load segmentation masks', type=str2bool, nargs='?', const=True)
    # parser.add_argument('--face_mask', default=False, help='Generate a mask for predicting only the face area', type=str2bool, nargs='?', const=True)
    # parser.add_argument('--use_grabcut', default=True, help='Use grabcut algorithm on the face mask to better segment the foreground', type=str2bool, nargs='?', const=True)
    # parser.add_argument('--scale_mask', default=1.4, help='Look over a wider section of foreground for grabcut', type=float)
    # parser.add_argument('--composite_mask', default=True, help='Merge the unmasked area back into the generated image', type=str2bool, nargs='?', const=True)
    # parser.add_argument('--composite_blur', default=8, help='Size of blur filter to smoothly composite the images', type=int)

    # Video params
    parser.add_argument('--video_dir', default='videos', help='Directory for storing training videos')
    parser.add_argument('--output_video', default=False, help='Generate videos of the optimization process', type=str2bool)
    parser.add_argument('--video_codec', default='MJPG', help='FOURCC-supported video codec name')
    parser.add_argument('--video_frame_rate', default=24, help='Video frames per second', type=int)
    parser.add_argument('--video_size', default=512, help='Video size in pixels', type=int)
    parser.add_argument('--video_skip', default=1, help='Only write every n frames (1 = write every frame)', type=int)

    args, other_args = parser.parse_known_args()


    # Extract FRS embds
    config = yaml.load(open(args.frs_config_path))
    images = tf.placeholder(dtype=tf.float32, shape=[None, config['image_size'], config['image_size'], 3], name='input_image')
    train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase')
    train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_last')
    embds, _ = get_embd(images, train_phase_dropout, train_phase_bn, config)
    print('Preparing FRS embeddings...')
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    variables_to_restore = slim.get_variables_to_restore(include=["embd_extractor"])
    with tf.Session(config=tf_config) as sess:
        tf.global_variables_initializer().run()
        print('Loading FRS model...')
        saver = tf.train.Saver(var_list=variables_to_restore)
        # saver=tf.train.import_meta_graph("config_ms1m_100_1006k/best-m-1006000.meta",clear_devices=True)
        saver.restore(sess, args.frs_model_path)
        print('Done!')
        batch_size = config['batch_size']
        imgs, _, fns = load_image_frs(args.src_dir, config['image_size'])
        print('Extracting embeddings...')
        embds_arr = run_embds(sess, imgs, batch_size, config['image_size'], 0, embds, images, train_phase_dropout, train_phase_bn)
        embds_arr = embds_arr/np.linalg.norm(embds_arr, axis=1, keepdims=True)
        print('Done!')
        frs_embds_dict = dict(zip(fns, list(embds_arr)))
    tf.reset_default_graph()

    ref_images=[]
    with open(args.morph_list_CSV, newline='') as csvfile:
        print('reading data: ' + args.morph_list_CSV)
        line = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in line:
            if row:
                ref_images.append(row)



    # Encoding
    args.decay_steps *= 0.01 * args.iterations # Calculate steps as a percent of total iterations

    if args.output_video:
      import cv2
      synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=1)

    # ref_images = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    # ref_images = list(filter(os.path.isfile, ref_images))

    if len(ref_images) == 0:
        raise Exception('%s is empty' % args.src_dir)

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)
    os.makedirs(args.generated_images_dir, exist_ok=True)
    os.makedirs(args.dlatent_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)

    # Initialize generator and perceptual model

    tflib.init_tf()
    
    # Local model
    url = os.path.abspath(args.model_url)
    with open(url, 'rb') as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)
    
    # Url model
    # with dnnlib.util.open_url(args.model_url, cache_dir=config.cache_dir) as f:
    #     generator_network, discriminator_network, Gs_network = pickle.load(f)


    generator = Generator(Gs_network, 1, clipping_threshold=args.clipping_threshold, tiled_dlatent=args.tile_dlatents, model_res=args.model_res, randomize_noise=args.randomize_noise)
    if (args.dlatent_avg != ''):
        generator.set_dlatent_avg(np.load(args.dlatent_avg))

    perc_model = None
    # if (args.use_lpips_loss > 0.00000001):
    #     # Url model
    #     with dnnlib.util.open_url('https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2', cache_dir=config.cache_dir) as f:
    #         perc_model =  pickle.load(f)
        # Local model
        # with open('models/vgg16_zhang_perceptual.pkl', 'rb') as f:
        #     perc_model =  pickle.load(f)
    perceptual_model = PerceptualModel(args, frs_embds_dict=frs_embds_dict, perc_model=perc_model, batch_size=1)
    perceptual_model.build_perceptual_model(generator, discriminator_network)

    ff_model = None

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch in tqdm(split_to_batches(ref_images, 1), total=len(ref_images)//1):
        names=[]
        name_temp=[]
        for pair in images_batch:
            for filename in pair:
                name_temp.append(filename.split('.')[0])
            names.append('-vs-'.join(name_temp))
        if args.output_video:
          video_out = {}
          for name in names:
            video_out[name] = cv2.VideoWriter(os.path.join(args.video_dir, f'{name}.avi'),cv2.VideoWriter_fourcc(*args.video_codec), args.video_frame_rate, (args.video_size,args.video_size))
        
        perceptual_model.set_reference_images(images_batch, names[0])
        dlatents = None
        if (args.load_last != ''): # load previous dlatents for initialization
            for name in names:
                dl = np.expand_dims(np.load(os.path.join(args.load_last, f'{name}.npy')),axis=0)
                if (dlatents is None):
                    dlatents = dl
                else:
                    dlatents = np.vstack((dlatents,dl))
        else:
            if (ff_model is None):
                if os.path.exists(args.load_resnet):
                    from keras.applications.resnet50 import preprocess_input
                    print(" ")
                    print("Loading ResNet Model:")
                    ff_model = load_model(args.load_resnet)
            if (ff_model is not None): # predict initial dlatents with ResNet model
                if (args.use_preprocess_input):
                    dlatents = (0.5*ff_model.predict(preprocess_input(load_ref_image(os.path.join(args.src_dir,images_batch[0][0]),image_size=args.resnet_image_size)))+0.5*ff_model.predict(preprocess_input(load_ref_image(os.path.join(args.src_dir,images_batch[0][1]),image_size=args.resnet_image_size))))
                else:
                    dlatents = (0.5*ff_model.predict(load_ref_image(os.path.join(args.src_dir,images_batch[0][0]),image_size=args.resnet_image_size))+0.5*ff_model.predict(load_ref_image(os.path.join(args.src_dir,images_batch[0][1]),image_size=args.resnet_image_size)))
        if dlatents is not None:
            generator.set_dlatents(dlatents)
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=args.iterations, use_optimizer=args.optimizer)
        pbar = tqdm(op, leave=False, total=args.iterations)
        vid_count = 0
        best_loss = None
        best_dlatent = None
        avg_loss_count = 0
        if args.early_stopping:
            avg_loss = prev_loss = None
        for loss_dict in pbar:
            if args.early_stopping: # early stopping feature
                if prev_loss is not None:
                    if avg_loss is not None:
                        avg_loss = 0.5 * avg_loss + (prev_loss - loss_dict["loss"])
                        if avg_loss < args.early_stopping_ratio and prev_loss<args.early_stopping_threshold: # count while under threshold; else reset
                            avg_loss_count += 1
                        else:
                            avg_loss_count = 0
                        if avg_loss_count > args.early_stopping_patience: # stop once threshold is reached
                            print("")
                            break
                    else:
                        avg_loss = prev_loss - loss_dict["loss"]
            pbar.set_description(" ".join(names) + ": " + "; ".join(["{} {:.4f}".format(k, v) for k, v in loss_dict.items()]))
            if best_loss is None or loss_dict["loss"] < best_loss:
                if best_dlatent is None or args.average_best_loss <= 0.00000001:
                    best_dlatent = generator.get_dlatents()
                else:
                    best_dlatent = 0.25 * best_dlatent + 0.75 * generator.get_dlatents()
                if args.use_best_loss:
                    generator.set_dlatents(best_dlatent)
                best_loss = loss_dict["loss"]
            if args.output_video and (vid_count % args.video_skip == 0):
              batch_frames = generator.generate_images()
              for i, name in enumerate(names):
                video_frame = PIL.Image.fromarray(batch_frames[i], 'RGB').resize((args.video_size,args.video_size),PIL.Image.LANCZOS)
                video_out[name].write(cv2.cvtColor(np.array(video_frame).astype('uint8'), cv2.COLOR_RGB2BGR))
            if args.clipping_threshold>0:    
                generator.stochastic_clip_dlatents()
            prev_loss = loss_dict["loss"]
        if not args.use_best_loss:
            best_loss = prev_loss
        print(" ".join(names), " Loss {:.4f}".format(best_loss))

        if args.output_video:
            for name in names:
                video_out[name].release()

        # Generate images from found dlatents and save them
        if args.use_best_loss:
            generator.set_dlatents(best_dlatent)
        generated_images = generator.generate_images()
        generated_dlatents = generator.get_dlatents()
        for img_array, dlatent, img_path, img_name in zip(generated_images, generated_dlatents, images_batch, names):
            # mask_img = None
            # if args.composite_mask and (args.load_mask or args.face_mask):
            #     _, im_name = os.path.split(img_path)
            #     mask_img = os.path.join(args.mask_dir, f'{im_name}')
            # if args.composite_mask and mask_img is not None and os.path.isfile(mask_img):
            #     orig_img = PIL.Image.open(img_path).convert('RGB')
            #     width, height = orig_img.size
            #     imask = PIL.Image.open(mask_img).convert('L').resize((width, height))
            #     imask = imask.filter(ImageFilter.GaussianBlur(args.composite_blur))
            #     mask = np.array(imask)/255
            #     mask = np.expand_dims(mask,axis=-1)
            #     img_array = mask*np.array(img_array) + (1.0-mask)*np.array(orig_img)
            #     img_array = img_array.astype(np.uint8)
                #img_array = np.where(mask, np.array(img_array), orig_img)
            img = PIL.Image.fromarray(img_array, 'RGB')
            img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
            np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), dlatent)
        generator.reset_dlatents()


if __name__ == "__main__":
    main()
