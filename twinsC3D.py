import tensorflow as tf
import img_input
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '3'}

keras = tf.keras
layers = keras.layers
models = keras.models
optimizers = keras.optimizers
Input = keras.Input
Model = keras.Model
Callbacks = keras.callbacks  # 回调函数
plot_model = keras.utils.plot_model
regularizers = keras.regularizers
board_path = 'board'

class_num = 20

Callbacks_list = [
    # Callbacks.EarlyStopping(monitor='acc',patience=5),
    Callbacks.ModelCheckpoint(
        filepath='my_model.h5', monitor='val-acc', save_best_only=True, mode='max'),
    Callbacks.TensorBoard(log_dir=board_path, histogram_freq=1)
]  # 回调函数配置


# 网络配置
# 卷积层（孪生层）
def vgg_16_base(input_tensor):
    #input_tensor = Input(shape=(200, 100,3))
    x = layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=(200, 100, 3))(input_tensor)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    out_tensor = layers.Flatten()(x)
    return out_tensor


# Cnn_layers = Model(input_tensor, out_tensor)


# def vgg_16_base(input_tensor):
#     # net = layers.Conv2D(1,(1,1),activation='relu',padding='same',input_shape=(200, 100,3))(input_tensor)
#     net = layers.Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(200, 100,3))(input_tensor)
#     net = layers.Conv2D(64,(3,3),activation='relu',padding='same')(net)
#     net = layers.MaxPooling2D((2,2),strides=(2,2))(net)

#     net = layers.Conv2D(128,(3,3),activation='relu',padding='same')(net)
#     net = layers.Conv2D(128,(3,3),activation='relu',padding='same')(net)
#     net= layers.MaxPooling2D((2,2),strides=(2,2))(net)

#     net = layers.Conv2D(256,(3,3),activation='relu',padding='same')(net)
#     net = layers.Conv2D(256,(3,3),activation='relu',padding='same')(net)
#     net = layers.Conv2D(256,(3,3),activation='relu',padding='same')(net)
#     net = layers.MaxPooling2D((2,2),strides=(2,2))(net)

#     net = layers.Conv2D(512,(3,3),activation='relu',padding='same')(net)
#     net = layers.Conv2D(512,(3,3),activation='relu',padding='same')(net)
#     net = layers.Conv2D(512,(3,3),activation='relu',padding='same')(net)
#     net = layers.MaxPooling2D((2,2),strides=(2,2))(net)

#     net = layers.Conv2D(512,(3,3),activation='relu',padding='same')(net)
#     net = layers.Conv2D(512,(3,3),activation='relu',padding='same')(net)
#     net = layers.Conv2D(512,(3,3),activation='relu',padding='same')(net)
#     net = layers.MaxPooling2D((2,2),strides=(2,2))(net)
#     net = layers.Flatten()(net)
#     return net


input_tensor = Input(shape=(200, 100, 3))
Cnn_layers = Model(input_tensor, vgg_16_base(input_tensor))

# 三输入
left_input = Input(shape=(200, 100, 3))
left_out = Cnn_layers(left_input)

right_input = Input(shape=(200, 100, 3))
right_out = Cnn_layers(right_input)

dif_input = Input(shape=(200, 100, 3))
dif_out = Cnn_layers(dif_input)

# 融合特征+全连接
merged = layers.concatenate([left_out, right_out, dif_out])
merged = layers.Dense(512, activation='relu',
                      kernel_regularizer=regularizers.l2(0.001))(merged)
merged = layers.Dropout(0.5)(merged)
merged = layers.Dense(512, activation='relu')(merged)
merged = layers.Dropout(0.5)(merged)
merged = layers.Dense(512, activation='relu')(merged)
merged = layers.Dropout(0.5)(merged)
predictions = layers.Dense(class_num, activation='softmax')(merged)
model = Model([left_input, right_input, dif_input], predictions, name='out')

model.summary()
model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['acc'])


if __name__ == "__main__":
    # 绘制网络结构图
    plot_model(Cnn_layers, show_shapes=True, to_file='Cnn_layers.png')
    plot_model(model, show_shapes=True, to_file='model.png')

    # 训练
    history = model.fit_generator(img_input.my_generator(1, 85000, class_num=class_num),
                                  steps_per_epoch=4000,
                                  epochs=200,
                                  validation_data=img_input.my_generator(
                                      85001, 86000, class_num=class_num),
                                  validation_steps=20,
                                  callbacks=Callbacks_list)
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
