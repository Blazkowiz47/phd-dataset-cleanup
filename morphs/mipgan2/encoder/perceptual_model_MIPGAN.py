# __author__ = "Haoyu Zhang"
# __copyright__ = "Copyright (C) 2021 Norwegian University of Science and Technology"
# __license__ = "License Agreement provided by Norwegian University of Science and Technology (NTNU)" \
#              "(MIPGAN-license-210420.pdf)"
# __version__ = "1.0"

from __future__ import absolute_import, division, print_function, unicode_literals

import bz2
import os

import keras.backend as K
import numpy as np
import PIL.Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from mtcnn import MTCNN
from PIL import ImageFilter

from ..dnnlib import tflib as tflib


####################################################
def box2square(box):
    if box[2] == box[3]:
        return box
    else:
        center_x = box[0] + 0.5 * box[2]
        center_y = box[1] + 0.6 * box[3]
        if box[2] > box[3]:
            new_w = int(box[2])
            new_h = int(box[2])
            new_x = int(box[0])
            new_y = int(center_y - 0.5 * new_h)
        elif box[2] < box[3]:
            new_w = int(box[3])
            new_h = int(box[3])
            new_x = int(center_x - 0.5 * new_h)
            new_y = int(box[1])
        return [new_x, new_y, new_w, new_h]


def box2square_tight(box):
    if box[2] == box[3]:
        return box
    else:
        center_x = box[0] + 0.5 * box[2]
        center_y = box[1] + 0.6 * box[3]
        if box[2] < box[3]:
            new_w = int(box[2])
            new_h = int(box[2])
            new_x = int(box[0])
            new_y = int(center_y - 0.5 * new_h)
        elif box[2] > box[3]:
            new_w = int(box[3])
            new_h = int(box[3])
            new_x = int(center_x - 0.5 * new_h)
            new_y = int(box[1])
        return [new_x, new_y, new_w, new_h]


####################################################
# from scipy import misc
from .. import model
import yaml


def generator_output_2_arcface(
    generator_output_tensor,
    box,
    target_img_size=112,
):
    temp_tensor = tf.transpose(generator_output_tensor, [0, 2, 3, 1])
    cutted_tensor = tf.slice(
        temp_tensor, [0, box[0], box[1], 0], [1, box[2], box[3], 3]
    )
    reshaped_tensor = tf.reshape(
        tf.image.resize_nearest_neighbor(
            cutted_tensor, (target_img_size, target_img_size), align_corners=True
        ),
        [1, target_img_size, target_img_size, 3],
    )
    return reshaped_tensor


def distance_loss(embeddings1, embeddings2, distance_type="Cosine"):
    if distance_type == "Euclidian":
        # Euclidian distance
        embeddings1 = embeddings1 / tf.norm(embeddings1, axis=0, keepdims=True)
        embeddings2 = embeddings2 / tf.norm(embeddings2, axis=0, keepdims=True)
        diff = tf.subtract(embeddings1, embeddings2)
        dist = tf.reduce_sum(tf.square(diff), 0)
    elif distance_type == "Cosine":
        # Distance based on cosine similarity
        dot = tf.reduce_sum(tf.multiply(embeddings1, embeddings2), axis=0)
        norm = tf.norm(embeddings1, axis=0) * tf.norm(embeddings2, axis=0)
        similarity = tf.divide(dot, norm)
        similarity = tf.minimum(1.0, similarity)
        dist = tf.subtract(1.0, similarity)
        # dist = tf.acos(similarity) / math.pi
    else:
        raise "Undefined distance metric"

    return dist


##############################################################
def load_images(images_list, image_size=256, sharpen=False):
    loaded_images = list()
    for img_path in images_list:
        img = PIL.Image.open(img_path).convert("RGB")
        if image_size is not None:
            img = img.resize((image_size, image_size), PIL.Image.LANCZOS)
            if sharpen:
                img = img.filter(ImageFilter.DETAIL)
        img = np.array(img)
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    return loaded_images


def load_ref_image(path, image_size=256, sharpen=True):
    img = PIL.Image.open(path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size, image_size), PIL.Image.LANCZOS)
    if sharpen:
        img = img.filter(ImageFilter.DETAIL)
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img


def tf_custom_adaptive_loss(a, b):
    from adaptive import lossfun

    shape = a.get_shape().as_list()
    dim = np.prod(shape[1:])
    a = tf.reshape(a, [-1, dim])
    b = tf.reshape(b, [-1, dim])
    loss, _, _ = lossfun(b - a, var_suffix="1")
    return tf.math.reduce_mean(loss)


def tf_custom_adaptive_rgb_loss(a, b):
    from adaptive import image_lossfun

    loss, _, _ = image_lossfun(b - a, color_space="RGB", representation="PIXEL")
    return tf.math.reduce_mean(loss)


def tf_custom_l1_loss(img1, img2):
    return tf.math.reduce_mean(tf.math.abs(img2 - img1), axis=None)


def tf_custom_logcosh_loss(img1, img2):
    return tf.math.reduce_mean(tf.keras.losses.logcosh(img1, img2))


def tf_custom_mse_loss(img1, img2):
    return tf.math.reduce_mean(tf.losses.mean_squared_error(img1, img2))


def create_stub(batch_size):
    return tf.constant(0, dtype="float32", shape=(batch_size, 0))


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, "wb") as fp:
        fp.write(data)
    return dst_path


class PerceptualModel:
    def __init__(self, args, frs_embds_dict, batch_size=1, perc_model=None, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.epsilon = 0.00000001
        self.lr = args.lr
        self.src_dir = args.src_dir
        self.decay_rate = args.decay_rate
        self.decay_steps = args.decay_steps
        self.img_size = args.image_size
        # self.layer = args.use_vgg_layer
        # self.vgg_loss = args.use_vgg_loss
        # self.face_mask = args.face_mask
        # self.use_grabcut = args.use_grabcut
        # self.scale_mask = args.scale_mask
        # self.mask_dir = args.mask_dir
        self.perceptual_loss = args.use_perceptual_loss
        self.arcloss_weight = args.use_arcface_loss
        self.arcloss = None
        self.arc_diff_loss_weight = args.use_arcface_difference_loss
        self.arc_diff_loss = None
        if self.perceptual_loss <= self.epsilon:
            self.perceptual_loss = None
        # if (self.layer <= 0 or self.vgg_loss <= self.epsilon):
        #     self.vgg_loss = None
        self.pixel_loss_mse = args.use_pixel_loss_mse
        if self.pixel_loss_mse <= self.epsilon:
            self.pixel_loss_mse = None
        self.pixel_loss_logcosh = args.use_pixel_loss_logcosh
        if self.pixel_loss_logcosh <= self.epsilon:
            self.pixel_loss_logcosh = None
        self.msssim_loss = args.use_msssim_loss
        if self.msssim_loss <= self.epsilon:
            self.msssim_loss = None
        # self.lpips_loss = args.use_lpips_loss
        # if (self.lpips_loss <= self.epsilon):
        #     self.lpips_loss = None
        self.l1_penalty = args.use_l1_penalty
        if self.l1_penalty <= self.epsilon:
            self.l1_penalty = None
        if self.arcloss_weight <= self.epsilon:
            self.arcloss_weight = None
        if self.arc_diff_loss_weight <= self.epsilon:
            self.arc_diff_loss_weight = None
        # self.adaptive_loss = args.use_adaptive_loss
        self.sharpen_input = args.sharpen_input
        self.batch_size = batch_size
        # if perc_model is not None and self.lpips_loss is not None:
        #     self.perc_model = perc_model
        # else:
        #     self.perc_model = None
        self.ref_img = None
        self.morph_weight = None
        self.ref_img1 = None
        self.ref_img2 = None
        self.perceptual_model = None
        # self.ref_img_features = None
        # self.features_weight = None
        self.loss = None
        self.discriminator_loss = args.use_discriminator_loss
        if self.discriminator_loss <= self.epsilon:
            self.discriminator_loss = None
        if self.discriminator_loss is not None:
            self.discriminator = None
            self.stub = create_stub(batch_size)

        self.frs_model_path = args.frs_model_path
        self.frs_config_path = args.frs_config_path

        self.perceptual1_1 = None
        self.perceptual1_2 = None
        self.perceptual2_2 = None
        self.perceptual3_3 = None
        self.ref_img1_feature_1_1 = None
        self.ref_img1_feature_1_2 = None
        self.ref_img1_feature_2_2 = None
        self.ref_img1_feature_3_3 = None
        self.ref_img2_feature_1_1 = None
        self.ref_img2_feature_1_2 = None
        self.ref_img2_feature_2_2 = None
        self.ref_img2_feature_3_3 = None
        # if self.face_mask:
        #     import dlib
        #     self.detector = dlib.get_frontal_face_detector()
        #     LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        #     landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
        #                                             LANDMARKS_MODEL_URL, cache_subdir='temp'))
        #     self.predictor = dlib.shape_predictor(landmarks_model_path)
        self.embeddings = None
        self.ref_img_sub1 = None
        self.ref_img_sub2 = None
        self.arcface_config = None
        self.embds_dict = frs_embds_dict
        self.img_name = None
        # self.list_loss=[]
        # self.list_lr=[]
        # self.list_arcface_loss=[]
        # self.list_arcface_difference_loss=[]

    def add_placeholder(self, var_name):
        var_val = getattr(self, var_name)
        setattr(
            self,
            var_name + "_placeholder",
            tf.placeholder(var_val.dtype, shape=var_val.get_shape()),
        )
        setattr(
            self,
            var_name + "_op",
            var_val.assign(getattr(self, var_name + "_placeholder")),
        )

    def assign_placeholder(self, var_name, var_val):
        self.sess.run(
            getattr(self, var_name + "_op"),
            {getattr(self, var_name + "_placeholder"): var_val},
        )

    def build_perceptual_model(self, generator, discriminator=None):
        ##################################################################################################################

        generated_image_tensor = generator.generated_image
        generated_image = tf.image.resize_nearest_neighbor(
            generated_image_tensor, (self.img_size, self.img_size), align_corners=True
        )
        ####shape#########################################
        # generator.generator_output: 1,3,1024,1024
        # self.generated_image_arcface: 1,112,112,3
        #################################################
        detector = MTCNN()
        box = detector.detect_faces(
            generated_image_tensor.eval().reshape((1024, 1024, 3))
        )[0]["box"]
        box = box2square(box)
        self.generated_image_arcface = generator_output_2_arcface(
            generator.generator_output, box, 112
        )

        # FRS embeddings for each contributing subjects
        self.ref_embedding_sub1 = tf.get_variable(
            "ref_embedding_sub1",
            shape=(512,),
            dtype="float32",
            initializer=tf.initializers.zeros(),
        )
        self.ref_embedding_sub2 = tf.get_variable(
            "ref_embedding_sub2",
            shape=(512,),
            dtype="float32",
            initializer=tf.initializers.zeros(),
        )
        self.add_placeholder("ref_embedding_sub1")
        self.add_placeholder("ref_embedding_sub2")

        # Load FRS model
        self.arcface_config = yaml.load(open(self.frs_config_path))
        embeddings_morph, _ = model.get_embd(
            self.generated_image_arcface, False, False, self.arcface_config
        )
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True

        variables_to_restore = slim.get_variables_to_restore(include=["embd_extractor"])
        saver = tf.train.Saver(var_list=variables_to_restore)
        saver.restore(self.sess, self.frs_model_path)

        # Embeddings of morphed image
        embeddings_morph = tf.reshape(embeddings_morph, self.ref_embedding_sub1.shape)

        # Identity-loss
        if self.arcloss_weight is not None:
            self.arcloss = (
                distance_loss(embeddings_morph, self.ref_embedding_sub1) * 0.5
                + distance_loss(embeddings_morph, self.ref_embedding_sub2) * 0.5
            )

        # Identity-Difference loss
        if self.arc_diff_loss_weight is not None:
            self.arc_diff_loss = tf.abs(
                distance_loss(embeddings_morph, self.ref_embedding_sub1)
                - distance_loss(embeddings_morph, self.ref_embedding_sub2)
            )

        ##################################################################################################################
        # VGG model for Perceptual-loss
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))

        # Learning rate
        global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="global_step"
        )
        incremented_global_step = tf.assign_add(global_step, 1)
        self._reset_global_step = tf.assign(global_step, 0)
        self.learning_rate = tf.train.exponential_decay(
            self.lr,
            incremented_global_step,
            self.decay_steps,
            self.decay_rate,
            staircase=True,
        )
        self.sess.run([self._reset_global_step])

        if self.discriminator_loss is not None:
            self.discriminator = discriminator

        self.morph_weight = tf.get_variable(
            "morph_weight",
            shape=generated_image.shape,
            dtype="float32",
            initializer=tf.initializers.zeros(),
        )
        self.add_placeholder("morph_weight")

        self.ref_img1 = tf.get_variable(
            "ref_img1",
            shape=generated_image.shape,
            dtype="float32",
            initializer=tf.initializers.zeros(),
        )
        self.ref_img2 = tf.get_variable(
            "ref_img2",
            shape=generated_image.shape,
            dtype="float32",
            initializer=tf.initializers.zeros(),
        )
        self.add_placeholder("ref_img1")
        self.add_placeholder("ref_img2")

        if self.perceptual_loss is not None:
            self.perceptual1_1 = Model(
                inputs=vgg16.input, outputs=vgg16.get_layer("block1_conv1").output
            )
            self.perceptual1_1.trainable = False
            img_feature_1_1 = self.perceptual1_1(
                preprocess_input(self.morph_weight * generated_image)
            )

            self.perceptual1_2 = Model(
                inputs=vgg16.input, outputs=vgg16.get_layer("block1_conv2").output
            )
            self.perceptual1_2.trainable = False
            img_feature_1_2 = self.perceptual1_2(
                preprocess_input(self.morph_weight * generated_image)
            )

            self.perceptual2_2 = Model(
                inputs=vgg16.input, outputs=vgg16.get_layer("block2_conv2").output
            )
            self.perceptual2_2.trainable = False
            img_feature_2_2 = self.perceptual2_2(
                preprocess_input(self.morph_weight * generated_image)
            )

            self.perceptual3_3 = Model(
                inputs=vgg16.input, outputs=vgg16.get_layer("block3_conv3").output
            )
            self.perceptual3_3.trainable = False
            img_feature_3_3 = self.perceptual3_3(
                preprocess_input(self.morph_weight * generated_image)
            )

            ## Layers used in the original perceptual loss
            # self.perceptual1_1 = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block1_conv1').output)
            # self.perceptual1_1.trainable = False
            # img_feature_1_1=self.perceptual1_1(preprocess_input(self.morph_weight * generated_image))

            # self.perceptual1_2 = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block2_conv2').output)
            # self.perceptual1_2.trainable = False
            # img_feature_1_2=self.perceptual1_2(preprocess_input(self.morph_weight * generated_image))

            # self.perceptual2_2 = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block3_conv3').output)
            # self.perceptual2_2.trainable = False
            # img_feature_2_2=self.perceptual2_2(preprocess_input(self.morph_weight * generated_image))

            # self.perceptual3_3 = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block4_conv3').output)
            # self.perceptual3_3.trainable = False
            # img_feature_3_3=self.perceptual3_3(preprocess_input(self.morph_weight * generated_image))

            self.ref_img1_feature_1_1 = tf.get_variable(
                "ref_img1_feature_1_1",
                shape=img_feature_1_1.shape,
                dtype="float32",
                initializer=tf.initializers.zeros(),
            )
            self.ref_img1_feature_1_2 = tf.get_variable(
                "ref_img1_feature_1_2",
                shape=img_feature_1_2.shape,
                dtype="float32",
                initializer=tf.initializers.zeros(),
            )
            self.ref_img1_feature_2_2 = tf.get_variable(
                "ref_img1_feature_2_2",
                shape=img_feature_2_2.shape,
                dtype="float32",
                initializer=tf.initializers.zeros(),
            )
            self.ref_img1_feature_3_3 = tf.get_variable(
                "ref_img1_feature_3_3",
                shape=img_feature_3_3.shape,
                dtype="float32",
                initializer=tf.initializers.zeros(),
            )
            self.add_placeholder("ref_img1_feature_1_1")
            self.add_placeholder("ref_img1_feature_1_2")
            self.add_placeholder("ref_img1_feature_2_2")
            self.add_placeholder("ref_img1_feature_3_3")
            self.ref_img2_feature_1_1 = tf.get_variable(
                "ref_img2_feature_1_1",
                shape=img_feature_1_1.shape,
                dtype="float32",
                initializer=tf.initializers.zeros(),
            )
            self.ref_img2_feature_1_2 = tf.get_variable(
                "ref_img2_feature_1_2",
                shape=img_feature_1_2.shape,
                dtype="float32",
                initializer=tf.initializers.zeros(),
            )
            self.ref_img2_feature_2_2 = tf.get_variable(
                "ref_img2_feature_2_2",
                shape=img_feature_2_2.shape,
                dtype="float32",
                initializer=tf.initializers.zeros(),
            )
            self.ref_img2_feature_3_3 = tf.get_variable(
                "ref_img2_feature_3_3",
                shape=img_feature_3_3.shape,
                dtype="float32",
                initializer=tf.initializers.zeros(),
            )
            self.add_placeholder("ref_img2_feature_1_1")
            self.add_placeholder("ref_img2_feature_1_2")
            self.add_placeholder("ref_img2_feature_2_2")
            self.add_placeholder("ref_img2_feature_3_3")

            loss_perceptual = (
                tf_custom_mse_loss(self.ref_img1_feature_1_1, img_feature_1_1)
                + tf_custom_mse_loss(self.ref_img1_feature_1_2, img_feature_1_2)
                + tf_custom_mse_loss(self.ref_img1_feature_2_2, img_feature_2_2)
                + tf_custom_mse_loss(self.ref_img1_feature_3_3, img_feature_3_3)
                + tf_custom_mse_loss(self.ref_img2_feature_1_1, img_feature_1_1)
                + tf_custom_mse_loss(self.ref_img2_feature_1_2, img_feature_1_2)
                + tf_custom_mse_loss(self.ref_img2_feature_2_2, img_feature_2_2)
                + tf_custom_mse_loss(self.ref_img2_feature_3_3, img_feature_3_3)
            )

        self.loss = 0
        # Traditional Perceptuaal Loss
        if self.perceptual_loss is not None:
            self.loss += self.perceptual_loss * loss_perceptual * 0.5

        # + logcosh loss on image pixels
        if self.pixel_loss_logcosh is not None:
            self.loss += (
                self.pixel_loss_logcosh
                * tf_custom_logcosh_loss(self.ref_img1, generated_image)
                * 0.5
            )
            self.loss += (
                self.pixel_loss_logcosh
                * tf_custom_logcosh_loss(self.ref_img2, generated_image)
                * 0.5
            )

        # + mse loss on image pixels
        if self.pixel_loss_mse is not None:
            self.loss += (
                self.pixel_loss_mse
                * tf_custom_mse_loss(self.ref_img1, generated_image)
                * 0.5
            )
            self.loss += (
                self.pixel_loss_mse
                * tf_custom_mse_loss(self.ref_img2, generated_image)
                * 0.5
            )

        # + MS-SIM loss on image pixels
        if self.msssim_loss is not None:
            self.loss += (
                self.msssim_loss
                * tf.math.reduce_mean(
                    1 - tf.image.ssim_multiscale(self.ref_img1, generated_image, 1)
                )
                * 0.5
            )
            self.loss += (
                self.msssim_loss
                * tf.math.reduce_mean(
                    1 - tf.image.ssim_multiscale(self.ref_img2, generated_image, 1)
                )
                * 0.5
            )

        # + L1 penalty on dlatent weights
        if self.l1_penalty is not None:
            self.loss += (
                self.l1_penalty
                * 512
                * tf.math.reduce_mean(
                    tf.math.abs(
                        generator.dlatent_variable - generator.get_dlatent_avg()
                    )
                )
            )
        # discriminator loss (realism)
        if self.discriminator_loss is not None:
            self.loss += self.discriminator_loss * tf.math.reduce_mean(
                self.discriminator.get_output_for(
                    tflib.convert_images_from_uint8(
                        generated_image_tensor, nhwc_to_nchw=True
                    ),
                    self.stub,
                )
            )
        # NOTICE: check whether saturating loss is used in the training of StyleGAN

        if self.arcloss_weight is not None:
            self.loss += self.arcloss_weight * self.arcloss

        if self.arc_diff_loss_weight is not None:
            self.loss += self.arc_diff_loss * self.arc_diff_loss_weight

    # def generate_face_mask(self, im):
    #     from imutils import face_utils
    #     import cv2
    #     rects = self.detector(im, 1)
    #     # loop over the face detections
    #     for (j, rect) in enumerate(rects):
    #         """
    #         Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
    #         """
    #         shape = self.predictor(im, rect)
    #         shape = face_utils.shape_to_np(shape)

    #         vertices = cv2.convexHull(shape)
    #         mask = np.zeros(im.shape[:2],np.uint8)
    #         cv2.fillConvexPoly(mask, vertices, 1)
    #         if self.use_grabcut:
    #             bgdModel = np.zeros((1,65),np.float64)
    #             fgdModel = np.zeros((1,65),np.float64)
    #             rect = (0,0,im.shape[1],im.shape[2])
    #             (x,y),radius = cv2.minEnclosingCircle(vertices)
    #             center = (int(x),int(y))
    #             radius = int(radius*self.scale_mask)
    #             mask = cv2.circle(mask,center,radius,cv2.GC_PR_FGD,-1)
    #             cv2.fillConvexPoly(mask, vertices, cv2.GC_FGD)
    #             cv2.grabCut(im,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    #             mask = np.where((mask==2)|(mask==0),0,1)
    #         return mask

    def set_reference_images(self, images_list, name):
        subnames = images_list[0]
        self.img_name = name
        assert len(images_list) != 0 and len(images_list) <= self.batch_size
        ref_img1_feature_1_1 = None
        ref_img1_feature_1_2 = None
        ref_img1_feature_2_2 = None
        ref_img1_feature_3_3 = None
        ref_img2_feature_1_1 = None
        ref_img2_feature_1_2 = None
        ref_img2_feature_2_2 = None
        ref_img2_feature_3_3 = None

        ref_img1 = load_ref_image(os.path.join(self.src_dir, subnames[0]))
        ref_img2 = load_ref_image(os.path.join(self.src_dir, subnames[1]))

        self.assign_placeholder("ref_img1", ref_img1)
        self.assign_placeholder("ref_img2", ref_img2)

        image_mask = np.ones(self.morph_weight.shape)

        if self.perceptual_loss is not None:
            ref_img1_feature_1_1 = self.perceptual1_1.predict_on_batch(
                preprocess_input(image_mask * np.array(ref_img1))
            )
            ref_img1_feature_1_2 = self.perceptual1_2.predict_on_batch(
                preprocess_input(image_mask * np.array(ref_img1))
            )
            ref_img1_feature_2_2 = self.perceptual2_2.predict_on_batch(
                preprocess_input(image_mask * np.array(ref_img1))
            )
            ref_img1_feature_3_3 = self.perceptual3_3.predict_on_batch(
                preprocess_input(image_mask * np.array(ref_img1))
            )
            ref_img2_feature_1_1 = self.perceptual1_1.predict_on_batch(
                preprocess_input(image_mask * np.array(ref_img2))
            )
            ref_img2_feature_1_2 = self.perceptual1_2.predict_on_batch(
                preprocess_input(image_mask * np.array(ref_img2))
            )
            ref_img2_feature_2_2 = self.perceptual2_2.predict_on_batch(
                preprocess_input(image_mask * np.array(ref_img2))
            )
            ref_img2_feature_3_3 = self.perceptual3_3.predict_on_batch(
                preprocess_input(image_mask * np.array(ref_img2))
            )

        if image_mask is not None:
            self.assign_placeholder("morph_weight", image_mask)
        if (
            (ref_img1_feature_1_1 is not None)
            and (ref_img1_feature_1_2 is not None)
            and (ref_img1_feature_2_2 is not None)
            and (ref_img1_feature_3_3 is not None)
            and (ref_img2_feature_1_1 is not None)
            and (ref_img2_feature_1_2 is not None)
            and (ref_img2_feature_2_2 is not None)
            and (ref_img2_feature_3_3 is not None)
        ):
            self.assign_placeholder("ref_img1_feature_1_1", ref_img1_feature_1_1)
            self.assign_placeholder("ref_img1_feature_1_2", ref_img1_feature_1_2)
            self.assign_placeholder("ref_img1_feature_2_2", ref_img1_feature_2_2)
            self.assign_placeholder("ref_img1_feature_3_3", ref_img1_feature_3_3)
            self.assign_placeholder("ref_img2_feature_1_1", ref_img2_feature_1_1)
            self.assign_placeholder("ref_img2_feature_1_2", ref_img2_feature_1_2)
            self.assign_placeholder("ref_img2_feature_2_2", ref_img2_feature_2_2)
            self.assign_placeholder("ref_img2_feature_3_3", ref_img2_feature_3_3)
        self.assign_placeholder("morph_weight", image_mask)

        self.assign_placeholder("ref_embedding_sub1", self.embds_dict[subnames[0]])
        self.assign_placeholder("ref_embedding_sub2", self.embds_dict[subnames[1]])

        # tenboard_dir = 'tensorboard/'
        # writer = tf.summary.FileWriter(tenboard_dir)
        # writer.add_graph(self.sess.graph)

        self.list_loss = []
        self.list_lr = []
        self.list_arcface_loss = []
        self.list_arcface_difference_loss = []

    def optimize(self, vars_to_optimize, iterations=200, use_optimizer="adam"):
        vars_to_optimize = (
            vars_to_optimize
            if isinstance(vars_to_optimize, list)
            else [vars_to_optimize]
        )
        if use_optimizer == "lbfgs":
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                self.loss,
                var_list=vars_to_optimize,
                method="L-BFGS-B",
                options={"maxiter": iterations},
            )
        else:
            if use_optimizer == "ggt":
                optimizer = tf.contrib.opt.GGTOptimizer(
                    learning_rate=self.learning_rate
                )
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
            self.sess.run(tf.variables_initializer(optimizer.variables()))
            if self.arcloss is not None and self.arc_diff_loss is not None:
                fetch_ops = [
                    min_op,
                    self.loss,
                    self.learning_rate,
                    self.arcloss,
                    self.arc_diff_loss,
                ]
            elif self.arcloss is not None:
                fetch_ops = [min_op, self.loss, self.learning_rate, self.arcloss]
            else:
                fetch_ops = [min_op, self.loss, self.learning_rate]
        self.sess.run(self._reset_global_step)

        # self.sess.graph.finalize()  # Graph is read-only after this statement.
        for _ in range(iterations):
            if use_optimizer == "lbfgs":
                optimizer.minimize(self.sess, fetches=[vars_to_optimize, self.loss])
                yield {"loss": self.loss.eval()}
            else:
                if self.arcloss is not None and self.arc_diff_loss is not None:
                    _, loss, lr, arcface_loss, arcface_difference_loss = self.sess.run(
                        fetch_ops
                    )
                    # self.list_loss.append(loss)
                    # self.list_lr.append(lr)
                    # self.list_arcface_loss.append(arcface_loss)
                    # self.list_arcface_difference_loss.append(arcface_difference_loss)
                    yield {
                        "loss": loss,
                        "lr": lr,
                        "id_loss": self.arcloss_weight * arcface_loss,
                        "id_diff_loss": self.arc_diff_loss_weight
                        * arcface_difference_loss,
                    }
                elif self.arcloss is not None:
                    _, loss, lr, arcface_loss = self.sess.run(fetch_ops)
                    # self.list_loss.append(loss)
                    # self.list_lr.append(lr)
                    # self.list_arcface_loss.append(arcface_loss)
                    yield {
                        "loss": loss,
                        "lr": lr,
                        "id_loss": self.arcloss_weight * arcface_loss,
                    }
                else:
                    _, loss, lr = self.sess.run(fetch_ops)
                    yield {"loss": loss, "lr": lr}
