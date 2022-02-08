#importing libraries
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import tensorflow as tf
import imageio
import numpy as np
import tensorboard
from huggingface_hub import push_to_hub_keras

#necessary step to add for, if you want to train the model on GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

#setting constants
batch_size = 8
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128





