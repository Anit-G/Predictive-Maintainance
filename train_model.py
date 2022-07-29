import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense , LSTM, Dropout, Activation
import tensorflow_probability as tfp 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from Helper import logger
from Helper import r2_keras

def LSTM_model_1(seq_array, label_array, sequence_length,epoch=1000):

    logger.info('Define LSTM model 1 for Sequnced Data') 
    nb_features = seq_array.shape[2]
    nb_out = label_array.shape[1]
    logger.info(f"Feature Length: {nb_features}, Label Length: {nb_out}")

    model = Sequential()
    model.add(LSTM(
             input_shape=(sequence_length, nb_features),
             units=15,
             return_sequences=True))
    model.add(LSTM(
              units=15,
              return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(
              units=15,
              return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(
              units=15,
              return_sequences=False))
    model.add(Dense(units=nb_out))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae',r2_keras])
    
    logger.info('Model Compilation')
    print(model.summary())
    logger.info('Fitting LSTM model')
    logger.info(f"Sequence Array: {seq_array.shape}")
    logger.info(f"Label Array: {label_array.shape}")

    return model


def LSTM_model_2(seq_array, label_array, sequence_length,epoch=1000):

    logger.info('Define LSTM model 2 for Sequnced Data') 
    nb_features = seq_array.shape[2]
    nb_out = label_array.shape[1]
    logger.info(f"Feature Length: {nb_features}, Label Length: {nb_out}")
    num_components = 3   # Number of components in the mixture (2 would be optional, but most of the time we don't know)
    event_shape = [nb_out]   # shape of the target (10 steps)
    params_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)

    model = Sequential()
    model.add(LSTM(
             input_shape=(sequence_length, nb_features),
             units=25,
             return_sequences=True))
    model.add(LSTM(
              units=25,
              return_sequences=True))
    # model.add(Dropout(0.1))
    model.add(LSTM(
              units=25,
              return_sequences=True))
    model.add(LSTM(
              units=25,
              return_sequences=False))
    # model.add(LSTM(params_size,return_sequences=False, activation=None))
    model.add(Dense(30,activation='relu'))
    model.add(Dense(params_size,activation=None))
    model.add(tfp.layers.MixtureNormal(num_components, event_shape))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae',r2_keras])

    logger.info('Model Compilation')
    print(model.summary())
    logger.info('Fitting LSTM model')
    logger.info(f"Sequence Array: {seq_array.shape}")
    logger.info(f"Label Array: {label_array.shape}")

    return model


def LSTM_model_3(seq_array, label_array, sequence_length,epoch=1000):

    logger.info('Define LSTM model 1 for Sequnced Data') 
    nb_features = seq_array.shape[2]
    nb_out = label_array.shape[1]
    logger.info(f"Feature Length: {nb_features}, Label Length: {nb_out}")

    model = Sequential()
    model.add(LSTM(
             input_shape=(sequence_length, nb_features),
             units=18,
             return_sequences=True))
    model.add(LSTM(
              units=18,
              return_sequences=True))
    # model.add(Dropout(0.1))
    model.add(LSTM(
              units=18,
              return_sequences=True))
    # model.add(Dropout(0.1))
    model.add(LSTM(
              units=18,
              return_sequences=False))
    model.add(Dense(18,activation='relu'))
    model.add(Dense(9,activation='relu'))
    model.add(Dense(units=nb_out,activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae',r2_keras])
    
    logger.info('Model Compilation')
    print(model.summary())
    logger.info('Fitting LSTM model')
    logger.info(f"Sequence Array: {seq_array.shape}")
    logger.info(f"Label Array: {label_array.shape}")

    return model
