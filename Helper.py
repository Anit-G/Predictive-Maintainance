
from tabnanny import verbose
import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVR,SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
from sqlalchemy import column

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense , LSTM, Dropout, Activation
import tensorflow_probability as tfp 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
import math
import time
from tqdm import tqdm
import logging

# Setup logger
logging.basicConfig(filename="logger.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
# Creating an object
logger = logging.getLogger('Helper')
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)
logger.info("---------- Start Log -----------")
# Setting seed for reproducibility
np.random.seed(0) 

def prepare_train_data(data, f = 0):
    """
    Function for creating RUL column using infromation form training set.
    RUL = max - time_in_cycles

    """
    logger.info(f"Data Preperation adding RUL column, factor: {f}")

    df = data.copy()
    fd_RUL = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    fd_RUL = pd.DataFrame(fd_RUL)
    fd_RUL.columns = ['unit_number','max']
    df = df.merge(fd_RUL, on=['unit_number'], how='left')
    df['RUL'] = df['max'] - df['time_in_cycles']
    df.drop(columns=['max'],inplace = True)

    logger.info(f"Data preped, Returning Data")
    return df[df['time_in_cycles'] > f]


def score(y_true,y_pred,a1=10,a2=13):
    """
    Error Fucntion using exponential function
    """
    logger.info('Exponential Score Calculation')
    score = 0
    d = y_pred - y_true
    for i in d:
        if i >= 0 :
            score += math.exp(i/a2) - 1   
        else:
            score += math.exp(- i/a1) - 1
    logger.info('Return Score')
    return score

"""Furthre scoring function using sklearn"""
def score_func(y_true,y_pred):
    logger.info('Score Calculations: MEA,RMSE,R2')
    lst = [round(score(y_true,y_pred),2), 
          round(mean_absolute_error(y_true,y_pred),2),
          round(mean_squared_error(y_true,y_pred),2)**0.5,
          round(r2_score(y_true,y_pred),2)]
    
    print(f' compatitive score {lst[0]}')
    print(f' mean absolute error {lst[1]}')
    print(f' root mean squared error {lst[2]}')
    print(f' R2 score {lst[3]}')
    logger.info('Return Scores')
    return [lst[1], round(lst[2],2), lst[3]*100]

def lstm_data_preprocessing(raw_train_data, raw_test_data, raw_RUL_data):
    logger.info('##### LSTM Preprocessing Beginning')

    train_df = raw_train_data
    truth_df = raw_RUL_data
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
    
    # TRAIN _______________________________________________________________________________________________________________________________
    
    # we will only make use of "label1" for binary classification, 
    # to answer the question: Is a specific engine going to fail within w1 cycles?
    logger.info('Setup train data')
    logger.info('Create additional data columns: Creating RUL Data Bins')

    w1 = 30
    w0 = 15
    train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0 )
    train_df['label2'] = train_df['label1']
    train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2
    logger.info(f'Bin Sizes: 0-{w0}, {w0}-{w1}, >{w1}')
    # MinMax Normalization
    logger.info('Data Normalization, Min-Max')

    train_df['cycle_norm'] = train_df['time_in_cycles']
    cols_normalize = train_df.columns.difference(['unit_number','time_in_cycles','RUL','label1','label2'])

    min_max_scaler = MinMaxScaler()

    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                                 columns=cols_normalize, 
                                 index=train_df.index)

    join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
    train_df = join_df.reindex(columns = train_df.columns)
    print("train_df:- ",train_df.head(2))
    print("\n")
    logger.info('Finish Setup train data')
    logger.info(f'Train Data Shape: {train_df.shape}')
    
    # TEST ________________________________________________________________________________________________________________________________
    
#    raw_test_data.drop(columns=['sensor21','sensor22','sensor8','sensor4','sensor3','sensor19','sensor13'],inplace=True)
    test_df = raw_test_data
    logger.info('Setup test data')
    logger.info('Data Normalization, Min-Max')
    if 'max' in test_df.columns.values:
        logger.info('Max column present in test dataset, Dropping Columns')
        test_df.drop(columns=['max'], inplace=True)
    else:
        logger.info('Max column NOT present in test dataset')
    # MinMax Normalization
    min_max_scaler = MinMaxScaler()
    test_df['cycle_norm'] = test_df['time_in_cycles']
    norm_test_df = pd.DataFrame(min_max_scaler.fit_transform(test_df[cols_normalize]), 
                                columns=cols_normalize, 
                                index=test_df.index)
    test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
    test_df = test_join_df.reindex(columns = test_df.columns)
    test_df = test_df.reset_index(drop=True)
    
    # We use the ground truth dataset to generate labels for the test data.
    # generate column max for test data
    logger.info('Prepare RUL for test data')
    logger.info('Calculate Totat Max RUL')
    max= pd.DataFrame(test_df.groupby('unit_number')['time_in_cycles'].max()).reset_index()
    max.columns = ['unit_number','max']
    truth_df.columns = ['more']
    truth_df['unit_number'] = truth_df.index + 1
    truth_df['max'] = max['max'] + truth_df['more'] # adding true-rul value + max cycle of test data = total max RUL
    truth_df.drop('more', axis=1, inplace=True)

    # generate RUL for test data
    logger.info('Calculate RUL for Test Data')
    test_df = test_df.merge(truth_df, on=['unit_number'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['time_in_cycles']
    test_df.drop('max', axis=1, inplace=True) 

    # generate label columns w0 and w1 for test data
    logger.info('Create additional data columns, Creat RUL Data Bins')

    test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )
    test_df['label2'] = test_df['label1']
    test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2
    print("test_df:- ", test_df.head(2))
    logger.info(f'Bin Sizes: 0-{w0}, {w0}-{w1}, >{w1}')

    logger.info('Finish Setup test data')
    logger.info(f'Test Data Shape: {test_df.shape}')

    # function to reshape features into (samples, time steps, features) 
    def gen_sequence(id_df, seq_length, seq_cols):
        """
        We will be reshaping the dataset into sequences of a certain lenght of time steps for each id
        """
        logger.info(f"Generating Sequence for ID: {id_df['unit_number'].values[0]}")
        logger.info(f'Data sequencing: Sequence Length:{seq_length}, Sequence Cols: {seq_cols}')
        
        # for one id I put all the rows in a single matrix
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        # Iterate over two lists in parallel.
        # For example id1 has 192 rows and sequence_length is 50
        # Create sequence of length 50
        # 0-50, 1-51, 2-52 ... 111 191
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield data_matrix[start:stop, :]

    ## pick a large window size of 50 cycles
    sequence_length = 50


    # pick the feature columns
    logger.info('Define Sequence cols and Sequence length') 

    sequence_cols = list(test_df.columns[:-3])
    
    logger.info('Create Sequence validation list for unit_number 1') 
    # Create a Validation list using the the id=1 which has 192 rows, thus the list will have 
    # 192-50 = 142 elements each with shape of 50 rows and 25 features.
    val=list(gen_sequence(train_df[train_df['unit_number']==1], sequence_length, sequence_cols))
    print(f'Validation List Length: {len(val)}')
    logger.info(f'Validation Sequence Length: {len(val)}')

    # generator for the sequences
    # transform each id of the train dataset into a sequence 
    # Each element in the tuple is a list which sequences for every id
    logger.info('Creating Sequence for all train data')
    seq_gen = (list(gen_sequence(train_df[train_df['unit_number']==id], sequence_length, sequence_cols)) 
               for id in train_df['unit_number'].unique())
    # logger.info(f'Train Data Sequence: {seq_gen.shape}')
    # generate sequences and convert to numpy array converts (id,sequences,number of features) to (id*sequences, number of features)
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    print(f'Train Sequence Array Shape: {seq_array.shape}')

    # function to generate labels
    def gen_labels(id_df, seq_length, label):
        """
        Now that we have all the data sequenced we need to label it, the same way we generated sequences for each id
        this time we want to sequence the RUL instead, to get an RUL for each sequence

        For example: RUL-> list of 100 elements, sequence lenght 50
        
        Labeling = sequenced data -----> RUL[sequence lenght:total_lenght,:]
        """
        logger.info('Generate Labels for a given sequence')
        logger.info(f"Data Set ID: {id_df['unit_number'].values[0]}")

        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]
    
        return data_matrix[seq_length:num_elements, :]

    logger.info('Generate labels for train data') 
    # generate labels
    label_gen = [gen_labels(train_df[train_df['unit_number']==id], sequence_length, ['RUL']) 
                 for id in train_df['unit_number'].unique()]
    logger.info(f"Label Sequence: {len(label_gen)}")
    label_array = np.concatenate(label_gen).astype(np.float32)
    print(f'Label Array Shape: {label_array.shape}')
    print(f'Label Array: {label_array}')

    logger.info('##### LSTM preprocessing End')
    return seq_array, label_array, test_df, sequence_length, sequence_cols

def r2_keras(y_true, y_pred):
    """
    Coefficient of Determination 
    """
    logger.info('Manually Calculate R2 Score')
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    logger.info('Return Score')
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

from train_model import *


def lstm_train(seq_array, label_array, sequence_length,epoch=1000):
    model = LSTM_model_3(seq_array,label_array,sequence_length)
    # Fitting
    history = model.fit(seq_array, label_array, epochs=epoch, batch_size=300, validation_split=0.05, verbose=2)
            #   callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')])
            #   #keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)

    # Print the history
    print(f'History Keys: {history.history.keys()}')
         
    logger.info('Finished Training') 
    logger.info('Returning model and training history')
    return model, history

def lstm_test_evaluation_graphs(model, history, seq_array, label_array):
    logger.info('##### Plot History Metrics #####')

    logger.info('Plot R2 curve')
    # summarize history for R^2
    fig_acc = plt.figure(figsize=(10, 10))
    plt.plot(history.history['r2_keras'])
    plt.plot(history.history['val_r2_keras'])
    plt.title('model r^2')
    plt.ylabel('R^2')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    # fig_acc.savefig("model_r2.png")

    logger.info('Plot MAE curve')
    # summarize history for MAE
    fig_acc = plt.figure(figsize=(10, 10))
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model MAE')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    # fig_acc.savefig("model_mae.png")

    logger.info('Plot Loss curve')
    # summarize history for Loss
    fig_acc = plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    # fig_acc.savefig("model_regression_loss.png")

    # training metrics
    logger.info('Model Evaluation') 
    scores = model.evaluate(seq_array, label_array, verbose=0, batch_size=200)
    print('\nMAE: {}'.format(scores[1]))
    print('\nR^2: {}'.format(scores[2]))
    logger.info('Model Prediction')
    y_pred = model.predict(seq_array,verbose=0, batch_size=200)
    y_true = label_array

    test_set = pd.DataFrame(y_pred )
    test_set.head()
    # test_set.to_csv('submit_train.csv', index = None)

def lstm_valid_evaluation(lstm_test_df, model, sequence_length, sequence_cols):
    logger.info('##### LSTM Valid Evaluation #####')

    # We pick the last sequence for each id in the test data
    logger.info('Arrange Sequences') 

    seq_array_test_last = [lstm_test_df[lstm_test_df['unit_number']==id][sequence_cols].values[-sequence_length:] 
                           for id in lstm_test_df['unit_number'].unique() if len(lstm_test_df[lstm_test_df['unit_number']==id]) >= sequence_length]

    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

    # Similarly, we pick the labels
    logger.info('Pick Labels')

    y_mask = [len(lstm_test_df[lstm_test_df['unit_number']==id]) >= sequence_length for id in lstm_test_df['unit_number'].unique()]
    label_array_test_last = lstm_test_df.groupby('unit_number')['RUL'].nth(-1)[y_mask].values
    label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)

    estimator = model

    print(f'Test Sequence Array Shape: {seq_array_test_last.shape}')
    # test metrics
    print(f'Test Label Array Shape: {label_array_test_last.shape}')
    logger.info(f"Test Sequence: {seq_array_test_last.shape}")
    logger.info(f"Test Labels: {label_array_test_last.shape}")
    logger.info('Model Evaluation')

    scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last, verbose=0)
    print('\nMAE: {}'.format(scores_test[1]))
    print('\nR^2: {}'.format(scores_test[2]))

    logger.info('Model prediction')

    y_pred_test = estimator.predict(seq_array_test_last,verbose=0)
    y_true_test = label_array_test_last

    test_set = pd.DataFrame(y_pred_test)
    print(f'Pred set: {test_set.shape}')

    # Plot in blue color the predicted data and in green color the
    # actual data to verify visually the accuracy of the model.
    logger.info('Plot Data Prediction') 
    fig_verify = plt.figure(figsize=(10, 5))
    plt.plot(y_pred_test)
    plt.plot(y_true_test, color="orange")
    plt.title('prediction')
    plt.ylabel('value')
    plt.xlabel('row')
    plt.legend(['predicted', 'actual data'], loc='upper left')
    #plt.show()
    # fig_verify.savefig("model_regression_verify.png")

    logger.info('##### Finish LSTM Valid Evaluation #####')

    return scores_test[1], scores_test[2], y_pred_test

#function for creating and training models using the "Random forest" algorithm
def train_models(data,model = 'FOREST',epoch=1000):
    
    if model != 'LSTM':
        logger.info("Rearranging data for non-lstm models")
        # X = data.iloc[:,:14].to_numpy() 
        # Y = data.iloc[:,14:].to_numpy()
        # Y = np.ravel(Y)

        X = data.drop(columns=['RUL']).to_numpy()
        Y = data['RUL'].to_numpy()

    if model == 'FOREST':
        logger.info('Fitting Forest Model')
        logger.info(f"Data Shape, X: {X.shape}, Y: {Y.shape}")
         #  parameters for models are selected in a similar cycle, with the introduction 
         # of an additional param parameter into the function:
         #for i in range(1,11):
         #     xgb = train_models(train_df,param=i,model="XGB",)
         #     y_xgb_i_pred = xgb.predict(X_001_test)
         #     print(f'param = {i}')
         #     score_func(y_true,y_xgb_i_pred)
        model = RandomForestRegressor(n_estimators=70, max_features=7, max_depth=5, n_jobs=-1, random_state=1)
        model.fit(X,Y)
        logger.info("Forest Model Fitting Complete")
        logger.info("Returning Model")
        return model

    if model == 'LinR':
        logger.info('Fitting Linear Regression Model')
        logger.info(f"Data Shape, X: {X.shape}, Y: {Y.shape}")
        model = LinearRegression()
        model.fit(X,Y)
        logger.info("Linear Regression Model Fitting Complete")
        logger.info("Returning Model")
        return model

    if model == 'LSVM':
        logger.info('Fitting Linear SVM')
        logger.info(f"Data Shape, X: {X.shape}, Y: {Y.shape}")
        model = LinearSVR(random_state=1)
        model.fit(X,Y)
        logger.info("LinearSVM Model Fitting Complete")
        logger.info("Returning Model")
        return model

    if model == 'SVM':
        logger.info('Fitting SVM')
        logger.info(f"Data Shape, X: {X.shape}, Y: {Y.shape}")
        model = SVR()
        model.fit(X,Y)
        logger.info("SVM Fitting Complete")
        logger.info("Returning Model")
        return model

    if model == 'KNN':
        logger.info('Fitting KNN')
        logger.info(f"Data Shape, X: {X.shape}, Y: {Y.shape}")
        model = KNeighborsRegressor()
        model.fit(X,Y)
        logger.info("Logistic Regression Model Fitting Complete")
        logger.info("Returning Model")
        return model
    
    if model == 'GNB':
        logger.info('Fitting KNN')
        logger.info(f"Data Shape, X: {X.shape}, Y: {Y.shape}")
        model = GaussianNB()
        model.fit(X,Y)
        logger.info("Logistic Regression Model Fitting Complete")
        logger.info("Returning Model")
        return model
    
    if model == 'TREE':
        logger.info('Fitting Decision Tree Model')
        logger.info(f"Data Shape, X: {X.shape}, Y: {Y.shape}")
        model = DecisionTreeRegressor(max_features=7, max_depth=5, random_state=1)
        model.fit(X,Y)
        logger.info("Decision Tree Model Fitting Complete")
        logger.info("Returning Model")
        return model

    if model == 'CAT':
        logger.info('Fitting Cat Boosting Model')
        logger.info(f"Data Shape, X: {X.shape}, Y: {Y.shape}")
        model = CatBoostRegressor(iterations=500,learning_rate=0.1,loss_function='RMSE',max_depth=3,random_seed=1)
        model.fit(X,Y)
        logger.info("Cat Boosting Model Fitting Complete")
        logger.info("Returning Model")
        return model

    elif model == 'LSTM':
        logger.info('##### Running LSTM')
        seq_array, label_array, lstm_test_df, sequence_length, sequence_cols = lstm_data_preprocessing(data[0], data[1], data[2])
        model_instance, history = lstm_train(seq_array, label_array, sequence_length,epoch=epoch)
        logger.info('##### Finish LSTM') 
        return model_instance, history, lstm_test_df, seq_array, label_array, sequence_length, sequence_cols
            
    return

#function for joint display of real and predicted values

def plot_result(y_true,y_pred):
    logger.info("Plotting Results:-")
    logger.info(f"Dataset shape: {len(y_true)}")

    rcParams['figure.figsize'] = 12,10
    plt.plot(y_pred)
    plt.plot(y_true)
    plt.tick_params(axis='x', which='both', bottom=False, top=False,labelbottom=False)
    plt.ylabel('RUL')
    plt.xlabel('training samples')
    plt.legend(('Predicted', 'True'), loc='upper right')
    plt.title('COMPARISION OF Real and Predicted values')
    logger.info("Plotting Completed")
    #plt.show()
    return