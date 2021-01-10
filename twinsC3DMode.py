import tensorflow as tf

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


board_path = 'board'
weight_decay = 0.005
Batch = 20

Callbacks_list = [
    # Callbacks.EarlyStopping(monitor='acc',patience=5),
    Callbacks.ModelCheckpoint(
        filepath='my_model.h5', monitor='val-acc', save_best_only=True, mode='max'),
    Callbacks.TensorBoard(log_dir=board_path,
                          histogram_freq=1, embeddings_freq=1,)
]  # 回调函数配置

# 网络配置
# 卷积层（孪生层）

BATCH_SIZE = 1
EPOCHS_NUM = 50
CLIP_LENGTH = 16
CROP_SZIE = 112
CHANNEL_NUM = 3


def c3d_model(input_shape):
    x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(input_shape)
    x = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)

    x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    # x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
    #            activation='relu', kernel_regularizer=l2(weight_decay))(x)
    # x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    out_tensor = Flatten()(x)
    return out_tensor


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
