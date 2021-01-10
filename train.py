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
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    twinsC3DMode.model.compile(loss='binary_crossentropy',
                               optimizer=sgd, metrics=['accuracy'])
    twinsC3DMode.model.summary()
    # 绘制网络结构图
    plot_model(twinsC3DMode.Cnn_layers, show_shapes=True,
               to_file='Cnn_layers.png')
    plot_model(twinsC3DMode.model, show_shapes=True, to_file='model.png')
    # 训练

    history = twinsC3DMode.model.fit_generator(dataProcess.generator_train_batch(BATCH_SIZE),
                                               steps_per_epoch=int(
                                                   10/BATCH_SIZE),
                                               epochs=EPOCHS_NUM,
                                               validation_data=dataProcess.generator_validation_batch(
        BATCH_SIZE),
        validation_steps=BATCH_SIZE, callbacks=twinsC3DMode.Callbacks_list)


twinsC3DMode.model.save('.\h5\my_model.h5')
