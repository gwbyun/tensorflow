#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 19:04:07 2022

@author: gw
"""
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from model_Unet import build_unet
from metrics import dice_loss, dice_coef, iou
from keras.models import Model, load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import DataGenerator

config = ConfigProto()
config.gpu_options.allow_growth = True
sess = InteractiveSession(config=config)
K.set_session(sess)
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*6)])
  except RuntimeError as e:
    print(e)
'''
'''
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = InteractiveSession(config=config)
'''


H = 512
W = 512

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*.jpg")))
    y = sorted(glob(os.path.join(path, "mask", "*.jpg")))
    return x, y

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x/255.0
    x = x > 0.5
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    batch_size = 2
    lr = 1e-5
    num_epochs = 100
    
    model_path = os.path.join("files", "model_epoch200.h5")
    csv_path = os.path.join("files", "data.csv")

    """ Dataset """
    tartget_size = 150
    img_ch = 3
    num_class = 1
    

    train_generator = DataGenerator('data/train', train_labels['image'],
                                    train_labels['mask'],
                                    batch_size, tartget_size,
                                    img_ch, num_class)
    """ Model """
    #model = load_model(model_path)
    
    model = build_unet((H, W, 3))
    model.load_weights(model_path)
    

    metrics = [dice_coef, iou, Recall(), Precision()]
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['acc', 'mse'])

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=False),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-20, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        #EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks,
        shuffle=False
    )
    