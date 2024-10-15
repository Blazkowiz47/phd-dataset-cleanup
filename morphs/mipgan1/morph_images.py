import argparse
import csv
import os
import pickle
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import yaml
from imageio import imread
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from PIL import Image
from tqdm import tqdm

from .dnnlib import tflib as tflib
from .encoder.generator_model import Generator
from .encoder.perceptual_model_MIPGAN import PerceptualModel, load_ref_image
from .model import get_embd


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def distance(embeddings1, embeddings2, distance_type="Euclidian"):
    embeddings1 = embeddings1.astype(np.float64)
    embeddings2 = embeddings2.astype(np.float64)
    if distance_type == "Euclidian":
        # Euclidian distance
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=0, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=0, keepdims=True)
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 0)
    elif distance_type == "Cosine":
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=0)
        norm = np.linalg.norm(embeddings1, axis=0) * np.linalg.norm(embeddings2, axis=0)
        similarity = dot / norm
        similarity = min(1, similarity)
        dist = 1 - similarity
        # dist = np.arccos(similarity) / math.pi
    else:
        raise Exception("Undefined distance metric")

    return dist


def run_embds(
    sess,
    images,
    batch_size,
    image_size,
    train_mode,
    embds_ph,
    image_ph,
    train_ph_dropout,
    train_ph_bn,
):
    if train_mode >= 1:
        train = True
    else:
        train = False
    batch_num = len(images) // batch_size
    left = len(images) % batch_size
    embds = []
    for i in range(batch_num):
        image_batch = images[i * batch_size : (i + 1) * batch_size]
        cur_embd = sess.run(
            embds_ph,
            feed_dict={
                image_ph: image_batch,
                train_ph_dropout: train,
                train_ph_bn: train,
            },
        )
        embds += list(cur_embd)
    if left > 0:
        image_batch = np.zeros([batch_size, image_size, image_size, 3])
        image_batch[:left, :, :, :] = images[-left:]
        cur_embd = sess.run(
            embds_ph,
            feed_dict={
                image_ph: image_batch,
                train_ph_dropout: train,
                train_ph_bn: train,
            },
        )
        embds += list(cur_embd)[:left]
    return np.array(embds)


def load_image_frs(dir_path, image_size):
    if os.path.isdir(dir_path):
        paths = list(os.listdir(dir_path))
    else:
        paths = [dir_path]
    images = []
    images_f = []
    for path in tqdm(paths, desc="Loading frs"):
        img = imread(os.path.join(dir_path, path))
        img = np.array(Image.fromarray(img).resize((image_size, image_size)))
        # img = misc.imresize(img, [image_size, image_size])
        # img = img[s:s+image_size, s:s+image_size, :]
        img_f = np.fliplr(img)
        img = img / 127.5 - 1.0
        img_f = img_f / 127.5 - 1.0
        images.append(img)
        images_f.append(img_f)
    fns = [os.path.basename(p) for p in paths]
    print("done!")
    return (np.array(images), np.array(images_f), fns)


def driver(args: Tuple[int, str, str, str]) -> None:
    process_num, src_dir, morph_list_csv, output_dir = args

    frs_model_path = "./morphs/models/frs/config_ms1m_100_1006k/best-m-1006000"
    frs_config_path = "./morphs/models/frs/configs/config_ms1m_100.yaml"

    resnet_image_size = 256
    iterations = 150
    decay_steps = 6
    load_resnet = "./morphs/models/finetuned_resnet.h5"
    average_best_loss = 0.5
    optimizer = "adam"
    model_url = "./morphs/models/StyleGAN_finetuned_ICAO.pkl"
    model_res = 1024
    data_dir = "./morphs/mipgan1/data"
    randomize_noise = False
    tile_dlatents = False
    clipping_threshold = 0

    config = yaml.safe_load(open(frs_config_path))
    images = tf.placeholder(
        dtype=tf.float32,
        shape=[None, config["image_size"], config["image_size"], 3],
        name="input_image",
    )
    train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name="train_phase")
    train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name="train_phase_last")
    embds, _ = get_embd(images, train_phase_dropout, train_phase_bn, config)
    print("Preparing FRS embeddings...")
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    variables_to_restore = slim.get_variables_to_restore(include=["embd_extractor"])

    with tf.Session(config=tf_config) as sess:
        tf.global_variables_initializer().run()
        print("Loading FRS model...")
        saver = tf.train.Saver(var_list=variables_to_restore)
        # saver=tf.train.import_meta_graph("config_ms1m_100_1006k/best-m-1006000.meta",clear_devices=True)
        saver.restore(sess, frs_model_path)
        print("Done!")
        batch_size = config["batch_size"]
        imgs, _, fns = load_image_frs(src_dir, config["image_size"])
        print("Extracting embeddings...")
        embds_arr = run_embds(
            sess,
            imgs,
            batch_size,
            config["image_size"],
            0,
            embds,
            images,
            train_phase_dropout,
            train_phase_bn,
        )
        embds_arr = embds_arr / np.linalg.norm(embds_arr, axis=1, keepdims=True)
        print("Done!")
        frs_embds_dict = dict(zip(fns, list(embds_arr)))

    tf.reset_default_graph()

    ref_images = []
    with open(morph_list_csv, newline="") as csvfile:
        print("reading data: " + morph_list_csv)
        line = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in line:
            if row:
                ref_images.append(row)

    # Encoding
    decay_steps *= 0.01 * iterations  # Calculate steps as a percent of total iterations

    # ref_images = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    # ref_images = list(filter(os.path.isfile, ref_images))

    if len(ref_images) == 0:
        return

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize generator and perceptual model

    tflib.init_tf()

    print(os.path.isfile(model_url))
    # Local model
    url = os.path.abspath(model_url)
    with open(url, "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(
        Gs_network,
        1,
        clipping_threshold=clipping_threshold,
        tiled_dlatent=tile_dlatents,
        model_res=model_res,
        randomize_noise=randomize_noise,
    )
    ags = argparse.Namespace(
        src_dir=src_dir,
        generated_images_dir=output_dir,
        dlatent_dir="./latent_representations/",
        data_dir="./morphs/mipgan1/data",
        mask_dir="masks",
        loss_dir="loss",
        frs_config_path=frs_config_path,
        frs_model_path=frs_model_path,
        dlatent_avg="",
        model_url=model_url,
        model_res=1024,
        optimizer="adam",
        image_size=256,
        resnet_image_size=256,
        decay_rate=0.95,
        iterations=150,
        decay_steps=6,
        early_stopping=False,
        early_stopping_threshold=10000000,
        lr=0.03,
        early_stopping_ratio=0,
        early_stopping_patience=10,
        load_resnet=load_resnet,
        use_preprocess_input=True,
        use_best_loss=True,
        average_best_loss=0.5,
        sharpen_input=True,
        use_perceptual_loss=0.0002,
        use_pixel_loss_logcosh=0,
        use_pixel_loss_mse=0,
        use_msssim_loss=1,
        use_l1_penalty=0,
        use_discriminator_loss=0,
        use_arcface_loss=10,
        use_arcface_difference_loss=1,
        randomize_noise=False,
        tile_dlatents=False,
        clipping_threshold=0,
        video_dir="videos",
        output_video=False,
        video_codec="MJPG",
        video_frame_rate=24,
        video_size=512,
        video_skip=1,
    )
    perc_model = None
    perceptual_model = PerceptualModel(
        ags, frs_embds_dict=frs_embds_dict, perc_model=perc_model, batch_size=1
    )
    perceptual_model.build_perceptual_model(generator, discriminator_network)

    ff_model = None

    # Main loop
    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch in tqdm(
        split_to_batches(ref_images, 1),
        total=len(ref_images) // 1,
        position=process_num,
    ):
        names = []
        name_temp = []
        fileflag = False
        for pair in images_batch:
            for filename in pair:
                if not os.path.isfile(os.path.join(src_dir, filename)):
                    fileflag = True
                name_temp.append(filename.split(".")[0])
            names.append("-vs-".join(name_temp))

        if fileflag or os.path.isfile(os.path.join(output_dir, f"{names[0]}.png")):
            continue

        perceptual_model.set_reference_images(images_batch, names[0])
        dlatents = None

        if ff_model is None:
            ff_model = load_model(load_resnet)

        dlatents = 0.5 * ff_model.predict(
            preprocess_input(
                load_ref_image(
                    os.path.join(src_dir, images_batch[0][0]),
                    image_size=resnet_image_size,
                )
            )
        ) + 0.5 * ff_model.predict(
            preprocess_input(
                load_ref_image(
                    os.path.join(src_dir, images_batch[0][1]),
                    image_size=resnet_image_size,
                )
            )
        )

        generator.set_dlatents(dlatents)

        op = perceptual_model.optimize(
            generator.dlatent_variable,
            iterations=iterations,
            use_optimizer=optimizer,
        )
        pbar = tqdm(op, leave=False, total=iterations)
        best_loss = None
        best_dlatent = None
        for loss_dict in pbar:
            pbar.set_description(
                " ".join(names)
                + ": "
                + "; ".join(["{} {:.4f}".format(k, v) for k, v in loss_dict.items()])
            )
            if best_loss is None or loss_dict["loss"] < best_loss:
                if best_dlatent is None or average_best_loss <= 0.00000001:
                    best_dlatent = generator.get_dlatents()
                else:
                    best_dlatent = 0.25 * best_dlatent + 0.75 * generator.get_dlatents()

                generator.set_dlatents(best_dlatent)
                best_loss = loss_dict["loss"]

        print(" ".join(names), " Loss {:.4f}".format(best_loss))

        # Generate images from found dlatents and save them
        generator.set_dlatents(best_dlatent)
        generated_images = generator.generate_images()
        generated_dlatents = generator.get_dlatents()

        for img_array, _, _, img_name in zip(
            generated_images, generated_dlatents, images_batch, names
        ):
            img = Image.fromarray(img_array, "RGB")
            img.save(os.path.join(output_dir, f"{img_name}.png"), "PNG")

        generator.reset_dlatents()
