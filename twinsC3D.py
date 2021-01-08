import tensorflow as tf
import os
import train

keras = tf.keras
layers = keras.layers
models = keras.models
optimizers = keras.optimizers
SGD = optimizers.SGD
Input = keras.Input
Model = keras.Model
Callbacks = keras.callbacks  # 回调函数
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


board_path = 'board'
weight_decay = 0.005

# 网络配置
# 卷积层（孪生层）


def c3d_model(input_shape):
    x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(input_shape)
    x = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)

    x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    out_tensor = Flatten()(x)
    return out_tensor


BATCH_SIZE = 10
CLIP_LENGTH = 16
CROP_SZIE = 112
CHANNEL_NUM = 3
input_shape = (112, 112, 16, 3)

input_tensor = Input(shape=(CROP_SZIE,
                            CROP_SZIE, CLIP_LENGTH, CHANNEL_NUM))
Cnn_layers = Model(input_tensor, c3d_model(input_tensor))
# 双输入
left_input = Input(shape=(CROP_SZIE,
                          CROP_SZIE, CLIP_LENGTH, CHANNEL_NUM))
left_out = Cnn_layers(left_input)
right_input = Input(shape=(CROP_SZIE,
                           CROP_SZIE, CLIP_LENGTH, CHANNEL_NUM))
right_out = Cnn_layers(right_input)

# 融合特征+全连接
merged = layers.concatenate([left_out, right_out])
merged = Dense(4096, activation='relu',
               kernel_regularizer=l2(weight_decay))(merged)
merged = Dropout(0.5)(merged)
merged = Dense(4096, activation='relu',
               kernel_regularizer=l2(weight_decay))(merged)
merged = Dropout(0.5)(merged)
predictions = Dense(1, activation='sigmoid')(merged)
model = Model([left_input, right_input], predictions, name='out')

# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='sgd',
#     metrics=['acc'])
# model.summary()

if __name__ == "__main__":
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    model.summary()
    # 绘制网络结构图
    plot_model(Cnn_layers, show_shapes=True, to_file='Cnn_layers.png')
    plot_model(model, show_shapes=True, to_file='model.png')
    # 训练
    history = model.fit_generator(train.generator_train_batch(2),
                                  steps_per_epoch=200,
                                  epochs=200,
                                  validation_data=train.generator_train_batch(
                                      2),
                                  validation_steps=2)


# # 训练样品处理
# ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator

# train_datagen = ImageDataGenerator(rescale=1. / 255)
# train_generator = train_datagen.flow_from_directory(
#     'C:/Users/Song/Desktop/week/keras_test/kaggle/train_dir',
#     target_size=(150, 150),
#     batch_size=20,
#     class_mode='binary')
# validation_datagen = ImageDataGenerator(rescale=1. / 255)
# validation_generator = train_datagen.flow_from_directory(
#     'C:/Users/Song/Desktop/week/keras_test/kaggle/validation_dir',
#     target_size=(150, 150),
#     batch_size=20,
#     class_mode='binary')


# model.save('cat_and_dog_2.h5')
