# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras.layers.merge import concatenate, add
from tensorflow.keras.regularizers import l1, l2


def create_lstm_fusion_model(learning_rate,
                 num_nodes,dropout_rate, X_train_lin,l2reg):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_nodes:   Number of lstm nodes
    dropout_rate:         Drop-out value
   
    """

    lin_input = Input(shape=(X_train_lin.shape[1],))

    linsoftmax = Dense(2, activation='softmax')(lin_input)

    lstm_input = Input(shape=(20,5))
    layer1 = LSTM(units = num_nodes, return_sequences = False,
                  kernel_regularizer=l2(l2reg), bias_regularizer=l2(l2reg), 
                  recurrent_regularizer=l2(l2reg))(lstm_input) 

    l1ad = Dropout(dropout_rate)(layer1)
    nnsoftmax = Dense(2, activation='softmax')(l1ad)
        
    merge = add([linsoftmax, nnsoftmax])
    out = merge/2
    model = Model(inputs=[lin_input, lstm_input], outputs=out)

    model.compile(optimizer = RMSprop(learning_rate=learning_rate, clipvalue=0.5),
        loss = 'binary_crossentropy',
        metrics=['acc','binary_crossentropy', tf.keras.metrics.AUC(name='auc')]) 

    return model  


def create_gru_fusion_model(learning_rate,
                 num_nodes,dropout_rate, X_train_lin,l2reg):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_nodes:   Number of lstm nodes
    dropout_rate:         Drop-out value   
    """

    lin_input = Input(shape=(X_train_lin.shape[1],))

    linsoftmax = Dense(2, activation='softmax')(lin_input)

    lstm_input = Input(shape=(20,5))
    layer1 = GRU(units = num_nodes, return_sequences = False,
                  kernel_regularizer=l2(l2reg), bias_regularizer=l2(l2reg), 
                  recurrent_regularizer=l2(l2reg))(lstm_input) 

    l1ad = Dropout(dropout_rate)(layer1)
    nnsoftmax = Dense(2, activation='softmax')(l1ad)
        
    merge = add([linsoftmax, nnsoftmax])
    out = merge/2

    model = Model(inputs=[lin_input, lstm_input], outputs=out)

    model.compile(optimizer = RMSprop(learning_rate=learning_rate, clipvalue=0.5),
        loss = 'binary_crossentropy',
        metrics=['acc','binary_crossentropy', tf.keras.metrics.AUC(name='auc')])
    
    return model  



def create_ann_fusion_model(learning_rate,
                 num_nodes,dropout_rate, X_train_lin,l2reg):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_nodes:   Number of lstm nodes
    dropout_rate:         Drop-out value
    
    """


    lin_input = Input(shape=(X_train_lin.shape[1],))

    linsoftmax = Dense(2, activation='softmax')(lin_input)
    

    ann_input = Input(shape=(100,))

    layer1 = Dense(num_nodes,kernel_regularizer=l2(l2reg), activation='relu')(ann_input)
    l1da = Dropout(dropout_rate)(layer1)
    layer2 = Dense(int(num_nodes/2),kernel_regularizer=l2(l2reg), activation='relu')(l1da)
    
    l2da = Dropout(dropout_rate)(layer2)
    layer3 = Dense(num_nodes,kernel_regularizer=l2(l2reg), activation='relu')(l2da)

    
    l3da = Dropout(dropout_rate)(layer3)

    nnsoftmax = Dense(2, activation='softmax')(l3da)
        
    merge = add([linsoftmax, nnsoftmax])
    out = merge/2

    model = Model(inputs=[lin_input, ann_input], outputs=out)

    model.compile(optimizer = RMSprop(learning_rate=learning_rate, clipvalue=0.5),
        loss = 'binary_crossentropy',
        metrics=['acc','binary_crossentropy', tf.keras.metrics.AUC(name='auc')])
    

    return model  







def create_simple_lstm_model(learning_rate,
                 num_nodes,dropout_rate,l2reg):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_nodes:   Number of lstm nodes
    dropout_rate:         Drop-out value
    """


    lstm_input = Input(shape=(20,5))
    layer1 = LSTM(units = num_nodes, return_sequences = False)(lstm_input) 

    l1ad = Dropout(dropout_rate)(layer1)
    nnsoftmax = Dense(2, activation='softmax')(l1ad)
    
    model = Model(inputs=lstm_input, outputs=nnsoftmax)

    model.compile(optimizer = RMSprop(learning_rate=learning_rate, clipvalue=0.5),
        loss = 'binary_crossentropy',
        metrics=['acc','binary_crossentropy', tf.keras.metrics.AUC(name='auc')])

    return model  



def create_simple_gru_model(learning_rate,
                 num_nodes,dropout_rate,l2reg):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_nodes:   Number of lstm nodes
    dropout_rate:         Drop-out value
   
    """

    lstm_input = Input(shape=(20,5))
    layer1 = GRU(units = num_nodes, return_sequences = False)(lstm_input) 

    l1ad = Dropout(dropout_rate)(layer1)
    nnsoftmax = Dense(2, activation='softmax')(l1ad)
    
    model = Model(inputs=lstm_input, outputs=nnsoftmax)

    model.compile(optimizer = RMSprop(learning_rate=learning_rate, clipvalue=0.5),
        loss = 'binary_crossentropy',
        metrics=['acc','binary_crossentropy', tf.keras.metrics.AUC(name='auc')])

    return model  