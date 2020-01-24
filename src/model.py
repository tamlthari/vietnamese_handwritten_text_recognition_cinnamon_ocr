import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import Progbar

from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.constraints import MaxNorm

from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, MaxPooling2D, Reshape, MaxPool2D, Lambda
from tensorflow.keras.layers import multiply, Permute, Lambda, RepeatVector
from tensorflow.keras import applications
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt

class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []
        self.batch_val_losses = []
        self.batch_val_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        # reset_metrics: the metrics returned will be only for this batch. 
        # If False, the metrics will be statefully accumulated across batches.
        self.model.reset_metrics()
  
    def on_test_batch_end(self, batch, logs=None):
        self.batch_val_losses.append(logs['loss'])
        self.batch_val_acc.append(logs['acc'])
        # reset_metrics: the metrics returned will be only for this batch. 
        # If False, the metrics will be statefully accumulated across batches.
        self.model.reset_metrics()

def plot_stats(training_stats, val_stats, x_label='Training Steps', stats='loss'):
    stats, x_label = stats.title(), x_label.title()
    legend_loc = 'upper right' if stats=='loss' else 'lower right'
    training_steps = len(training_stats)
    test_steps = len(val_stats)

    plt.figure()
    plt.ylabel(stats)
    plt.xlabel(x_label)
    plt.plot(training_stats, label='Training ' + stats)
    plt.plot(np.linspace(0, training_steps, test_steps), val_stats, label='Validation ' + stats)
    plt.ylim([0,max(plt.ylim())])
    plt.legend(loc=legend_loc)
    plt.show()

callbacks = [
    TensorBoard(
        log_dir='./logs',
        histogram_freq=10,
        profile_batch=0,
        write_graph=True,
        write_images=False,
        update_freq="epoch"),
    ModelCheckpoint(
        filepath='checkpoint_weights.hdf5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1),
    EarlyStopping(
        monitor='val_loss',
        min_delta=1e-8,
        patience=15,
        restore_best_weights=True,
        verbose=1),
    ReduceLROnPlateau(
        monitor='val_loss',
        min_delta=1e-8,
        factor=0.2,
        patience=10,
        verbose=1)
]

def ctc_loss_lambda_func(y_true, y_pred):
    """Function for computing the CTC loss"""

    if len(y_true.shape) > 2:
        y_true = tf.squeeze(y_true)

    input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
    input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)
    label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")

    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    loss = tf.reduce_mean(loss)

    return loss

# def attention_rnn(inputs):
#     # inputs.shape = (batch_size, time_steps, input_dim)
#     input_dim = int(inputs.shape[3])
#     timestep = int(inputs.shape[2])
#     squeezed = Lambda(lambda x: K.squeeze(x, 1))(inputs)
#     a = Permute((2, 1))(squeezed)
#     a = Dense(timestep, activation='softmax')(a)
#     a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
#     a = RepeatVector(input_dim)(a)
#     a_probs = Permute((2, 1), name='attention_vec')(a)
#     output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
#     return output_attention_mul

def attention_rnn(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    timestep = int(inputs.shape[1])
    a = Permute((2, 1))(inputs)
    a = Dense(timestep, activation='softmax')(a)
    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

def maxpooling(base_model):
    model = Sequential(name='vgg16')
    for layer in base_model.layers[:-1]:
        if 'pool' in layer.name:
            pooling_layer = MaxPooling2D(pool_size=(2, 2), name=layer.name)
            model.add(pooling_layer)
        else:
            model.add(layer)
    return model

def build_model_quoc(input_size, d_model, learning_rate=3e-4):
    """
    Reference: attention layer as per Quoc. https://github.com/pbcquoc/vietnamese_ocr

    """

    input_data = Input(name='input', shape=input_size, dtype='float32')
    base_model = applications.VGG16(weights='imagenet', include_top=False)
    base_model = maxpooling(base_model)
    inner = base_model(input_data)

    #Adding attention
    shape = inner.get_shape()
    attn = Reshape((shape[1], shape[2] * shape[3]))(inner)

    # attn = Reshape(target_shape=(int(cnn.shape[1]), -1), name='reshape')(cnn)
    attn = Dense(512, activation='relu', kernel_initializer='he_normal', name='dense1')(attn) 
    attn = Dropout(0.25)(attn) 
    attn = attention_rnn(attn)


    blstm = Bidirectional(LSTM(units=256, return_sequences=True, kernel_initializer='he_normal', dropout=0.5))(attn)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, kernel_initializer='he_normal', dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, kernel_initializer='he_normal', dropout=0.5))(blstm)
#     blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
#     blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)

    blstm = Dropout(rate=0.5)(blstm)
    output_data = Dense(units=d_model, activation="softmax")(blstm)

#     optimizer = RMSprop(learning_rate=learning_rate)
    optimizer = Adam(learning_rate=learning_rate)
    
    model = Model(inputs=input_data, outputs=output_data)
    model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)

    return model


def build_model(input_size, d_model, learning_rate=3e-4):
    """
    Convolucional Recurrent Neural Network by Puigcerver et al.

    With attention layer as per Quoc. https://github.com/pbcquoc/vietnamese_ocr

    Reference:
        Joan Puigcerver.
        Are multidimensional recurrent layers really necessary for handwritten text recognition?
        In: Document Analysis and Recognition (ICDAR), 2017 14th
        IAPR International Conference on, vol. 1, pp. 67–72. IEEE (2017)

        Carlos Mocholí Calvo and Enrique Vidal Ruiz.
        Development and experimentation of a deep learning system for convolutional and recurrent neural networks
        Escola Tècnica Superior d’Enginyeria Informàtica, Universitat Politècnica de València, 2018
    """

    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same")(input_data)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    #Adding attention
    shape = cnn.get_shape()
    attn = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    # attn = Reshape(target_shape=(int(cnn.shape[1]), -1), name='reshape')(cnn)
    attn = Dense(512, activation='relu', kernel_initializer='he_normal', name='dense1')(attn) 
    attn = Dropout(0.25)(attn) 
    attn = attention_rnn(attn)


    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(attn)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
#     blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
#     blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)

    blstm = Dropout(rate=0.5)(blstm)
    output_data = Dense(units=d_model, activation="softmax")(blstm)

#     optimizer = RMSprop(learning_rate=learning_rate)
    optimizer = Adam(learning_rate=learning_rate)
    
    model = Model(inputs=input_data, outputs=output_data)
    model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)

    return model

def build_model_woattn(input_size, d_model, learning_rate=3e-4):
    """
    Convolucional Recurrent Neural Network by Puigcerver et al.

    Reference:
        Joan Puigcerver.
        Are multidimensional recurrent layers really necessary for handwritten text recognition?
        In: Document Analysis and Recognition (ICDAR), 2017 14th
        IAPR International Conference on, vol. 1, pp. 67–72. IEEE (2017)

        Carlos Mocholí Calvo and Enrique Vidal Ruiz.
        Development and experimentation of a deep learning system for convolutional and recurrent neural networks
        Escola Tècnica Superior d’Enginyeria Informàtica, Universitat Politècnica de València, 2018
    """

    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same")(input_data)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
#     blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
#     blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)

    blstm = Dropout(rate=0.5)(blstm)
    output_data = Dense(units=d_model, activation="softmax")(blstm)

#     optimizer = RMSprop(learning_rate=learning_rate)
    optimizer = Adam(learning_rate=learning_rate)
    
    model = Model(inputs=input_data, outputs=output_data)
    model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)

    return model

def build_model_(input_size, d_model, learning_rate=3e-4):
    """
    Convolucional Recurrent Neural Network by Puigcerver et al.

    Reference:
        Joan Puigcerver.
        Are multidimensional recurrent layers really necessary for handwritten text recognition?
        In: Document Analysis and Recognition (ICDAR), 2017 14th
        IAPR International Conference on, vol. 1, pp. 67–72. IEEE (2017)

        Carlos Mocholí Calvo and Enrique Vidal Ruiz.
        Development and experimentation of a deep learning system for convolutional and recurrent neural networks
        Escola Tècnica Superior d’Enginyeria Informàtica, Universitat Politècnica de València, 2018
    """

    input_data = Input(name="input", shape=input_size)

    conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(input_data)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
    
    conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
    
    conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
    batch_norm_3 = BatchNormalization()(conv_3)
    conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(batch_norm_3)
    pool_4 = MaxPool2D(pool_size=(2, 2))(conv_4)
    
    conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
    batch_norm_5 = BatchNormalization()(conv_5)
    conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 2))(batch_norm_6)
    
    conv_7 = Conv2D(512, (2,2), activation = 'relu', padding='same')(pool_6)
    batch_norm_7 = BatchNormalization()(conv_7)
    pool_7 = MaxPool2D(pool_size=(1, 4))(batch_norm_7) 
    
    shape = pool_7.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(pool_7)
    
    blstm = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.5))(blstm)
    blstm = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.5))(blstm)
    blstm = Dropout(rate=0.5)(blstm)
    output_data = Dense(d_model, activation = 'softmax')(blstm)

#     optimizer = RMSprop(learning_rate=learning_rate)
    optimizer = Adam(learning_rate=learning_rate)
    
    model = Model(inputs=input_data, outputs=output_data)
    model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)
    model.summary()
    return model

