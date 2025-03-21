"""
Trains a modified EfficientNet to generate approximate dlatents using examples from a trained StyleGAN.
Props to @SimJeg on GitHub for the original code this is based on, from this thread: https://github.com/Puzer/stylegan-encoder/issues/1#issuecomment-490469454
"""

import os
import math
import numpy as np
import pickle
import cv2
import argparse

from . import dnnlib
from .dnnlib import tflib as tflib
import config

import tensorflow
import keras.backend as K

from efficientnet import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    preprocess_input,
)

from keras.layers import (
    Input,
    LocallyConnected1D,
    Reshape,
    Permute,
    Conv2D,
    Add,
    Concatenate,
)
from keras.models import Model, load_model

"""
Truncation method from @oneiroid
"""


def truncate_fancy(
    dlat,
    dlat_avg,
    model_scale=18,
    truncation_psi=0.7,
    minlayer=0,
    maxlayer=8,
    do_clip=False,
):
    layer_idx = np.arange(model_scale)[np.newaxis, :, np.newaxis]
    ones = np.ones(layer_idx.shape, dtype=np.float32)
    coefs = np.where(layer_idx < maxlayer, truncation_psi * ones, ones)
    if minlayer > 0:
        coefs[0, :minlayer, :] = ones[0, :minlayer, :]
    if do_clip:
        return tflib.lerp_clip(dlat_avg, dlat, coefs).eval()
    else:
        return tflib.lerp(dlat_avg, dlat, coefs)


def truncate_normal(dlat, dlat_avg, truncation_psi=0.7):
    return (dlat - dlat_avg) * truncation_psi + dlat_avg


def generate_dataset_main(
    n=10000,
    save_path=None,
    seed=None,
    model_res=1024,
    image_size=256,
    minibatch_size=32,
    truncation=0.7,
    fancy_truncation=False,
):
    """
    Generates a dataset of 'n' images of shape ('size', 'size', 3) with random seed 'seed'
    along with their dlatent vectors W of shape ('n', 512)

    These datasets can serve to train an inverse mapping from X to W as well as explore the latent space

    More variation added to latents; also, negative truncation added to balance these examples.
    """

    n = n // 2  # this gets doubled because of negative truncation below
    model_scale = int(2 * (math.log(model_res, 2) - 1))  # For example, 1024 -> 18

    Gs = load_Gs()
    if model_scale % 3 == 0:
        mod_l = 3
    else:
        mod_l = 2
    if seed is not None:
        b = bool(np.random.RandomState(seed).randint(2))
        Z = np.random.RandomState(seed).randn(n * mod_l, Gs.input_shape[1])
    else:
        b = bool(np.random.randint(2))
        Z = np.random.randn(n * mod_l, Gs.input_shape[1])
    if b:
        mod_l = model_scale // 2
    mod_r = model_scale // mod_l
    if seed is not None:
        Z = np.random.RandomState(seed).randn(n * mod_l, Gs.input_shape[1])
    else:
        Z = np.random.randn(n * mod_l, Gs.input_shape[1])
    W = Gs.components.mapping.run(
        Z, None, minibatch_size=minibatch_size
    )  # Use mapping network to get unique dlatents for more variation.
    dlatent_avg = Gs.get_var("dlatent_avg")  # [component]
    if fancy_truncation:
        W = np.append(
            truncate_fancy(W, dlatent_avg, model_scale, truncation),
            truncate_fancy(W, dlatent_avg, model_scale, -truncation),
            axis=0,
        )
    else:
        W = np.append(
            truncate_normal(W, dlatent_avg, truncation),
            truncate_normal(W, dlatent_avg, -truncation),
            axis=0,
        )
    W = W[:, :mod_r]
    W = W.reshape((n * 2, model_scale, 512))
    X = Gs.components.synthesis.run(
        W,
        randomize_noise=False,
        minibatch_size=minibatch_size,
        print_progress=True,
        output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
    )
    X = np.array(
        [
            cv2.resize(x, (image_size, image_size), interpolation=cv2.INTER_AREA)
            for x in X
        ]
    )
    X = preprocess_input(X)
    return W, X


def generate_dataset(
    n=10000,
    save_path=None,
    seed=None,
    model_res=1024,
    image_size=256,
    minibatch_size=16,
    truncation=0.7,
    fancy_truncation=False,
):
    """
    Use generate_dataset_main() as a helper function.
    Divides requests into batches to save memory.
    """
    batch_size = 16
    inc = n // batch_size
    left = n - ((batch_size - 1) * inc)
    W, X = generate_dataset_main(
        inc,
        save_path,
        seed,
        model_res,
        image_size,
        minibatch_size,
        truncation,
        fancy_truncation,
    )
    for i in range(batch_size - 2):
        aW, aX = generate_dataset_main(
            inc,
            save_path,
            seed,
            model_res,
            image_size,
            minibatch_size,
            truncation,
            fancy_truncation,
        )
        W = np.append(W, aW, axis=0)
        aW = None
        X = np.append(X, aX, axis=0)
        aX = None
    aW, aX = generate_dataset_main(
        left,
        save_path,
        seed,
        model_res,
        image_size,
        minibatch_size,
        truncation,
        fancy_truncation,
    )
    W = np.append(W, aW, axis=0)
    aW = None
    X = np.append(X, aX, axis=0)
    aX = None

    if save_path is not None:
        prefix = "_{}_{}".format(seed, n)
        np.save(os.path.join(os.path.join(save_path, "W" + prefix)), W)
        np.save(os.path.join(os.path.join(save_path, "X" + prefix)), X)

    return W, X


def is_square(n):
    return n == int(math.sqrt(n) + 0.5) ** 2


def get_effnet_model(
    save_path,
    model_res=1024,
    image_size=256,
    depth=1,
    size=3,
    activation="elu",
    loss="logcosh",
    optimizer="adam",
):
    if os.path.exists(save_path):
        print("Loading model")
        return load_model(save_path)

    # Build model
    print("Building model")
    model_scale = int(2 * (math.log(model_res, 2) - 1))  # For example, 1024 -> 18
    if size <= 0:
        effnet = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
        )
    if size == 1:
        effnet = EfficientNetB1(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
        )
    if size == 2:
        effnet = EfficientNetB2(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
        )
    if size >= 3:
        effnet = EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
        )

    layer_size = model_scale * 8 * 8 * 8
    if is_square(layer_size):  # work out layer dimensions
        layer_l = int(math.sqrt(layer_size) + 0.5)
        layer_r = layer_l
    else:
        layer_m = math.log(math.sqrt(layer_size), 2)
        layer_l = 2 ** math.ceil(layer_m)
        layer_r = layer_size // layer_l
    layer_l = int(layer_l)
    layer_r = int(layer_r)

    x_init = None
    inp = Input(shape=(image_size, image_size, 3))
    x = effnet(inp)
    if size < 1:
        x = Conv2D(model_scale * 8, 1, activation=activation)(x)  # scale down
        if depth > 0:
            x = Reshape(
                (layer_r, layer_l)
            )(
                x
            )  # See https://github.com/OliverRichter/TreeConnect/blob/master/cifar.py - TreeConnect inspired layers instead of dense layers.
    else:
        if depth < 1:
            depth = 1
        if size <= 2:
            x = Conv2D(model_scale * 8 * 4, 1, activation=activation)(
                x
            )  # scale down a bit
            x = Reshape(
                (layer_r * 2, layer_l * 2)
            )(
                x
            )  # See https://github.com/OliverRichter/TreeConnect/blob/master/cifar.py - TreeConnect inspired layers instead of dense layers.
        else:
            x = Reshape((384, 256))(x)  # full size for B3
    while depth > 0:
        x = LocallyConnected1D(layer_r, 1, activation=activation)(x)
        x = Permute((2, 1))(x)
        x = LocallyConnected1D(layer_l, 1, activation=activation)(x)
        x = Permute((2, 1))(x)
        if x_init is not None:
            x = Add()([x, x_init])  # add skip connection
        x_init = x
        depth -= 1
    if (
        size >= 2
    ):  # add unshared layers at end for different sections of the latent space
        x_init = x
        if layer_r % 3 == 0 and layer_l % 3 == 0:
            a = LocallyConnected1D(layer_r, 1, activation=activation)(x)
            b = LocallyConnected1D(layer_r, 1, activation=activation)(x)
            c = LocallyConnected1D(layer_r, 1, activation=activation)(x)
            a = Permute((2, 1))(a)
            b = Permute((2, 1))(b)
            c = Permute((2, 1))(c)
            a = LocallyConnected1D(layer_l // 3, 1, activation=activation)(a)
            b = LocallyConnected1D(layer_l // 3, 1, activation=activation)(b)
            c = LocallyConnected1D(layer_l // 3, 1, activation=activation)(c)
            x = Concatenate()([a, b, c])
        else:
            a = LocallyConnected1D(layer_l, 1, activation=activation)(x)
            b = LocallyConnected1D(layer_l, 1, activation=activation)(x)
            a = Permute((2, 1))(a)
            b = Permute((2, 1))(b)
            a = LocallyConnected1D(layer_r // 2, 1, activation=activation)(a)
            b = LocallyConnected1D(layer_r // 2, 1, activation=activation)(b)
            x = Concatenate()([a, b])
        x = Add()([x, x_init])  # add skip connection
    x = Reshape((model_scale, 512))(x)  # train against all dlatent values
    model = Model(inputs=inp, outputs=x)
    model.compile(
        loss=loss, metrics=[], optimizer=optimizer
    )  # By default: adam optimizer, logcosh used for loss.
    return model


def finetune_effnet(model, args):
    """
    Finetunes an EfficientNet to predict W from X
    Generate batches (X, W) of size 'batch_size', iterates 'n_epochs', and repeat while 'max_patience' is reached
    on the test set. The model is saved every time a new best test loss is reached.
    """
    save_path = args.model_path
    model_res = args.model_res
    image_size = args.image_size
    batch_size = args.batch_size
    test_size = args.test_size
    max_patience = args.max_patience
    n_epochs = args.epochs
    seed = args.seed
    minibatch_size = args.minibatch_size
    truncation = args.truncation
    fancy_truncation = args.fancy_truncation
    use_ktrain = args.use_ktrain
    ktrain_max_lr = args.ktrain_max_lr
    ktrain_reduce_lr = args.ktrain_reduce_lr
    ktrain_stop_early = args.ktrain_stop_early

    assert image_size >= 224

    # Create a test set
    np.random.seed(seed)
    print("Creating test set:")
    W_test, X_test = generate_dataset(
        n=test_size,
        model_res=model_res,
        image_size=image_size,
        seed=seed,
        minibatch_size=minibatch_size,
        truncation=truncation,
        fancy_truncation=fancy_truncation,
    )

    # Iterate on batches of size batch_size
    print("Generating training set:")
    patience = 0
    epoch = -1
    best_loss = np.inf
    # loss = model.evaluate(X_test, W_test)
    # print('Initial test loss : {:.5f}'.format(loss))
    while patience <= max_patience:
        W_train = X_train = None
        W_train, X_train = generate_dataset(
            batch_size,
            model_res=model_res,
            image_size=image_size,
            seed=seed,
            minibatch_size=minibatch_size,
            truncation=truncation,
            fancy_truncation=fancy_truncation,
        )
        if use_ktrain:
            print("Creating validation set:")
            W_val, X_val = generate_dataset(
                n=test_size,
                model_res=model_res,
                image_size=image_size,
                seed=seed,
                minibatch_size=minibatch_size,
                truncation=truncation,
                fancy_truncation=fancy_truncation,
            )
            learner = ktrain.get_learner(
                model=model,
                train_data=(X_train, W_train),
                val_data=(X_val, W_val),
                workers=1,
                use_multiprocessing=False,
                batch_size=minibatch_size,
            )
            # learner.lr_find() # simulate training to find good learning rate
            # learner.lr_plot() # visually identify best learning rate
            learner.autofit(
                ktrain_max_lr,
                checkpoint_folder="/tmp",
                reduce_on_plateau=ktrain_reduce_lr,
                early_stopping=ktrain_stop_early,
            )
            learner = None
            print("Done with current validation set.")
            model.fit(
                X_val, W_val, epochs=n_epochs, verbose=True, batch_size=minibatch_size
            )
        else:
            model.fit(
                X_train,
                W_train,
                epochs=n_epochs,
                verbose=True,
                batch_size=minibatch_size,
            )
        loss = model.evaluate(X_test, W_test, batch_size=minibatch_size)
        if loss < best_loss:
            print("New best test loss : {:.5f}".format(loss))
            patience = 0
            best_loss = loss
        else:
            print("Test loss : {:.5f}".format(loss))
            patience += 1
        if (
            patience > max_patience
        ):  # When done with test set, train with it and discard.
            print("Done with current test set.")
            model.fit(
                X_test, W_test, epochs=n_epochs, verbose=True, batch_size=minibatch_size
            )
        print("Saving model.")
        model.save(save_path)


parser = argparse.ArgumentParser(
    description="Train an EfficientNet to predict latent representations of images in a StyleGAN model from generated examples",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--model_url",
    default="https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ",
    help="Fetch a StyleGAN model to train on from this URL",
)
parser.add_argument(
    "--model_res",
    default=1024,
    help="The dimension of images in the StyleGAN model",
    type=int,
)
parser.add_argument(
    "--data_dir", default="data", help="Directory for storing the EfficientNet model"
)
parser.add_argument(
    "--model_path",
    default="data/finetuned_effnet.h5",
    help="Save / load / create the EfficientNet model with this file path",
)
parser.add_argument(
    "--model_depth",
    default=1,
    help="Number of TreeConnect layers to add after EfficientNet",
    type=int,
)
parser.add_argument(
    "--model_size",
    default=1,
    help="Model size - 0 - small, 1 - medium, 2 - large, or 3 - full size.",
    type=int,
)
parser.add_argument(
    "--use_ktrain", default=False, help="Use ktrain for training", type=bool
)
parser.add_argument(
    "--ktrain_max_lr",
    default=0.001,
    help="Maximum learning rate for ktrain",
    type=float,
)
parser.add_argument(
    "--ktrain_reduce_lr",
    default=1,
    help="Patience for reducing learning rate after a plateau for ktrain",
    type=float,
)
parser.add_argument(
    "--ktrain_stop_early",
    default=3,
    help="Patience for early stopping for ktrain",
    type=float,
)
parser.add_argument(
    "--activation", default="elu", help="Activation function to use after EfficientNet"
)
parser.add_argument("--optimizer", default="adam", help="Optimizer to use")
parser.add_argument("--loss", default="logcosh", help="Loss function to use")
parser.add_argument(
    "--use_fp16", default=False, help="Use 16-bit floating point", type=bool
)
parser.add_argument(
    "--image_size", default=256, help="Size of images for EfficientNet model", type=int
)
parser.add_argument(
    "--batch_size",
    default=2048,
    help="Batch size for training the EfficientNet model",
    type=int,
)
parser.add_argument(
    "--test_size",
    default=512,
    help="Batch size for testing the EfficientNet model",
    type=int,
)
parser.add_argument(
    "--truncation",
    default=0.7,
    help="Generate images using truncation trick",
    type=float,
)
parser.add_argument(
    "--fancy_truncation",
    default=True,
    help="Use fancier truncation proposed by @oneiroid",
    type=float,
)
parser.add_argument(
    "--max_patience",
    default=2,
    help="Number of iterations to wait while test loss does not improve",
    type=int,
)
parser.add_argument(
    "--freeze_first",
    default=False,
    help="Start training with the pre-trained network frozen, then unfreeze",
    type=bool,
)
parser.add_argument(
    "--epochs",
    default=2,
    help="Number of training epochs to run for each batch",
    type=int,
)
parser.add_argument(
    "--minibatch_size",
    default=16,
    help="Size of minibatches for training and generation",
    type=int,
)
parser.add_argument(
    "--seed",
    default=-1,
    help="Pick a random seed for reproducibility (-1 for no random seed selected)",
    type=int,
)
parser.add_argument(
    "--loop",
    default=-1,
    help="Run this many iterations (-1 for infinite, halt with CTRL-C)",
    type=int,
)

args, other_args = parser.parse_known_args()

os.makedirs(args.data_dir, exist_ok=True)

if args.seed == -1:
    args.seed = None

if args.use_fp16:
    K.set_floatx("float16")
    K.set_epsilon(1e-4)

if args.use_ktrain:
    import ktrain

tflib.init_tf()

model = get_effnet_model(
    args.model_path,
    model_res=args.model_res,
    depth=args.model_depth,
    size=args.model_size,
    activation=args.activation,
    optimizer=args.optimizer,
    loss=args.loss,
)

with dnnlib.util.open_url(args.model_url, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)


def load_Gs():
    return Gs_network


# K.get_session().run(tensorflow.global_variables_initializer())

if args.freeze_first:
    model.layers[1].trainable = False
    model.compile(loss=args.loss, metrics=[], optimizer=args.optimizer)

model.summary()

if (
    args.freeze_first
):  # run a training iteration first while pretrained model is frozen, then unfreeze.
    finetune_effnet(model, args)
    model.layers[1].trainable = True
    model.compile(loss=args.loss, metrics=[], optimizer=args.optimizer)
    model.summary()

if args.loop < 0:
    while True:
        finetune_effnet(model, args)
else:
    count = args.loop
    while count > 0:
        finetune_effnet(model, args)
        count -= 1
