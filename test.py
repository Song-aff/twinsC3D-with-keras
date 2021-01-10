import tensorflow as tf
import os
import dataProcess
import twinsC3DMode
keras = tf.keras
layers = keras.layers
models = keras.models
optimizers = keras.optimizers
SGD = optimizers.SGD
Input = keras.Input
Model = keras.Model
Callbacks = keras.callbacks  # 回调函数 用于tensorborad
plot_model = keras.utils.plot_model
regularizers = keras.regularizers
l2 = regularizers.l2

Dense = layers.Dense
Dropout = layers.Dropout
Conv3D = layers.Conv3D
Input = layers.Input
MaxPool3D = layers.MaxPool3D
Flatten = layers.Flatten
Activation = layers.Activation

BATCH_SIZE = 2
EPOCHS_NUM = 2
CLIP_LENGTH = 16
CROP_SZIE = 112
CHANNEL_NUM = 3

if __name__ == "__main__":
    # history = twinsC3DMode.model.fit_generator(dataProcess.generator_train_batch(BATCH_SIZE),
    #                                            steps_per_epoch=int(
    #                                                10/BATCH_SIZE),
    #                                            epochs=EPOCHS_NUM,
    #                                            validation_data=dataProcess.generator_validation_batch(
    #     BATCH_SIZE),
    #     validation_steps=BATCH_SIZE, callbacks=twinsC3DMode.Callbacks_list)
 