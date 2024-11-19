#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 

"""
import os
import sys
import tensorflow as tf
import Networks.network as networks
from Networks import Losses
from Configuration.Configs import Variables
from Configuration.DataLoader import DataLoader

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")
v = Variables()

# %% Stage - 1: Define your data filepath and create a dataset
fp_input = r"/Users/jielu/Desktop/ORL_Seg/Dataset/Image"
fp_label = r"/Users/jielu/Desktop/ORL_Seg/Dataset/Mask"

train_image, train_label, valid_image, valid_label = DataLoader(
    image_fp=fp_input, label_fp=fp_label, valid_rate=0.2)()
v.print_config()
# %% Stage - 2: Load your model and train it.
model = networks.UNet(v.image_shape, out_cls=v.seg_num)(include_top=True)
model.compile(
    optimizer=tf.keras.optimizers.Adam(v.optimParas['learn_rate']),
    loss='categorical_crossentropy',  # todo: future add more loss function.
    metrics=['accuracy'],
)
history = model.fit(
    train_image, train_label,
    batch_size=v.fitParas['bs'], epochs=v.fitParas['epoch_sv'],
    validation_data=(valid_image, valid_label),
)

# %% If you want to save the model weights, uncomment the below codes.
# model.save_weights(model.name)
