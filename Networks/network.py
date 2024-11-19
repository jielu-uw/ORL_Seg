# -*- coding: utf-8 -*-
""" Objective: 
Created on (2022/08/02 - 11:58:04)

Script Descriptions: 

"""

# %% System Setup
import os
import sys
import tensorflow as tf
import numpy as np
# import Layers
import Networks.Layers as Layers
import math

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")


# TODO: PSPNet, DeepLabV3, 
class UNet:
    """
    https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28
    """

    def __init__(self, image_shape, out_cls):
        self.image_shape = image_shape
        self.out_cls = out_cls

    def __call__(self, include_top=True):
        Inputs = tf.keras.Input(self.image_shape)
        n_filters = [32, 32, 64, 64, 128]

        x = Layers.ConvBlock(use_BN=False)(Inputs)

        # 512 * 512 *64
        x = Layers.ConvBlock(fs=n_filters[0])(x)
        x = Layers.ConvBlock(fs=n_filters[0])(x)
        concat_1 = x
        x = tf.keras.layers.MaxPooling2D()(x)

        # 256 * 256 * 128
        x = Layers.ConvBlock(fs=n_filters[1])(x)
        x = Layers.ConvBlock(fs=n_filters[1])(x)
        concat_2 = x
        x = tf.keras.layers.MaxPooling2D()(x)

        # 128 * 128 * 256
        x = Layers.ConvBlock(fs=n_filters[2])(x)
        x = Layers.ConvBlock(fs=n_filters[2])(x)
        concat_3 = x
        x = tf.keras.layers.MaxPooling2D()(x)

        # 64 * 64 * 512
        x = Layers.ConvBlock(fs=n_filters[3])(x)
        x = Layers.ConvBlock(fs=n_filters[3])(x)
        concat_4 = x
        x = tf.keras.layers.MaxPooling2D()(x)

        # 32 * 32 * 1024
        x = Layers.ConvBlock(fs=n_filters[4])(x)
        x = Layers.ConvBlock(fs=n_filters[4])(x)

        x = Layers.TransposeConv(fs=n_filters[3])(x)  # 64 * 64 * 512
        x = tf.keras.layers.Concatenate()([x, concat_4])
        x = Layers.ConvBlock(fs=n_filters[3])(x)
        x = Layers.ConvBlock(fs=n_filters[3])(x)

        x = Layers.TransposeConv(fs=n_filters[2])(x)  # 128 * 128 * 256
        x = tf.keras.layers.Concatenate()([x, concat_3])
        x = Layers.ConvBlock(fs=n_filters[2])(x)
        x = Layers.ConvBlock(fs=n_filters[2])(x)

        x = Layers.TransposeConv(fs=n_filters[1])(x)  # 256 * 256 *128
        x = tf.keras.layers.Concatenate()([x, concat_2])
        x = Layers.ConvBlock(fs=n_filters[1])(x)
        x = Layers.ConvBlock(fs=n_filters[1])(x)

        x = Layers.TransposeConv(fs=n_filters[0])(x)  # 512 * 512 * 64
        x = tf.keras.layers.Concatenate()([x, concat_1])
        x = Layers.ConvBlock(fs=n_filters[0])(x)
        x = Layers.ConvBlock(fs=n_filters[0])(x)

        x = Layers.Conv(fs=self.out_cls, ks=1, s=1)(x)

        if include_top:
            if self.out_cls == 1:
                x = Layers.get_activation_layer("sigmoid")(x)
            elif self.out_cls > 1:
                x = Layers.get_activation_layer("softmax")(x)

        models = tf.keras.models.Model(inputs=Inputs, outputs=x, name="UNet")
        return models


class TransUNet:
    """https://arxiv.org/abs/2102.04306"""

    def __init__(self, image_shape, out_cls, num_heads=8):
        self.image_shape = image_shape
        self.num_heads = num_heads
        self.out_cls = out_cls

    def __call__(self, include_top=True):
        # Hit: ver2 use the [64, 128, 256, 512, 1024] for better performance.
        n_filters = [128, 256, 512, 1024]
        hidden_size = n_filters[-1]
        Inputs = tf.keras.Input(self.image_shape)

        x1 = Layers.ConvBlock(fs=n_filters[0], ks=7, s=2)(Inputs)  # 256*256*128
        x2 = Layers.ConvBlock(fs=n_filters[1], ks=3, s=2)(x1)  # 128*128*256
        x3 = Layers.ConvBlock(fs=n_filters[2], ks=3, s=2)(x2)  # 64*64*512
        x4 = Layers.ConvBlock(fs=n_filters[3], ks=3, s=2)(x3)  # 32*32*1024

        # Vision Transformer Block
        x = tf.reshape(tensor=x4, shape=(
            tf.shape(x4)[0], tf.shape(x4)[1] * tf.shape(x4)[2], tf.shape(x4)[3]
        ))
        drop_list = [x for x in np.linspace(0, 0.1, int(len(n_filters)))]
        for i in range(int(len(n_filters))):
            x = Layers.CCT_TransformerBlock(
                num_heads=self.num_heads, hidden_size=hidden_size,
                mlp_dims=hidden_size, factor=2, drop_prob=drop_list[i]
            )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.reshape(tensor=x, shape=(
            tf.shape(x4)[0], tf.shape(x4)[1], tf.shape(x4)[2], tf.shape(x4)[3]
        ))

        x = tf.keras.layers.Concatenate()([x, x4])
        x = Layers.TransposeConv(fs=n_filters[2])(x)  # 64, 64, 512

        x = tf.keras.layers.Concatenate()([x, x3])
        x = Layers.TransposeConv(fs=n_filters[1])(x)  # 128, 128, 256

        x = tf.keras.layers.Concatenate()([x, x2])
        x = Layers.TransposeConv(fs=n_filters[0])(x)  # 256, 256, 128

        x = tf.keras.layers.Concatenate()([x, x1])
        x = Layers.TransposeConv(fs=64)(x)
        x = Layers.ConvBlock(fs=64)(x)

        x = Layers.Conv(fs=self.out_cls)(x)
        if include_top:
            if self.out_cls == 1:
                x = Layers.get_activation_layer("sigmoid")(x)
            elif self.out_cls > 1:
                x = Layers.get_activation_layer("softmax")(x)

        models = tf.keras.models.Model(inputs=Inputs, outputs=x,
                                       name="TransUNet")
        return models





