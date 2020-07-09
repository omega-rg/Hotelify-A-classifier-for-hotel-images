import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# imagenet_utils.preprocess_input(x, data_format=data_format, mode='tf')


# tf.keras.preprocessing.image.ImageDataGenerator(
#     featurewise_center=False,
#     samplewise_center=False,
#     featurewise_std_normalization=False,
#     samplewise_std_normalization=False,
#     zca_whitening=False,
#     zca_epsilon=1e-06,
#     rotation_range=0,
#     width_shift_range=0.0,
#     height_shift_range=0.0,
#     brightness_range=None,
#     shear_range=0.0,
#     zoom_range=0.0,
#     channel_shift_range=0.0,
#     fill_mode="nearest",
#     cval=0.0,
#     horizontal_flip=False,
#     vertical_flip=False,
#     rescale=None,
#     preprocessing_function=None,
#     data_format=None,
#     validation_split=0.0,
#     dtype=None,
# )

# ImageDataGenerator.flow_from_directory(
#     directory,
#     target_size=(256, 256),
#     color_mode="rgb",
#     classes=None,
#     class_mode="categorical",
#     batch_size=32,
#     shuffle=True,
#     seed=None,
#     save_to_dir=None,
#     save_prefix="",
#     save_format="png",
#     follow_links=False,
#     subset=None,
#     interpolation="nearest",
# )

train_directory="Dataset/Train"
val_directory="Dataset/Val"
preprocessing_function= lambda x: preprocess_input(x,mode='tf')

train_datagen=ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.05,
    height_shift_range=0.1,
    brightness_range=[0.7,1],
    shear_range=10.0,
    zoom_range=[0.5,1.5],
    channel_shift_range=80.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=preprocessing_function,
    data_format=None,
    validation_split=0.0,
    dtype=None,
)

val_datagen=ImageDataGenerator(
    preprocessing_function=preprocessing_function
)

seed=1024

def train_generator():
    train_img_gen=train_datagen.flow_from_directory(
        train_directory,
        target_size=(256, 256),
        batch_size=32,
        shuffle=True,
        seed=seed,
    )
    return train_img_gen

def val_generator():
    val_img_gen=val_datagen.flow_from_directory(
        val_directory,
        target_size=(256, 256),
        batch_size=16,
        shuffle=True,
        seed=seed,
    )
    return val_img_gen
