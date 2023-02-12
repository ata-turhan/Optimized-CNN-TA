import time

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import Callback, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import (
    GRU,
    LSTM,
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from keras.models import Sequential
from tensorflow.keras.metrics import *


def create_model_MLP(activation_func="swish", dropout_rate=0.2, optimizer_algo="adam"):
    MLP = Sequential()
    MLP.add(
        Dense(
            64,
            input_shape=(30,),
            activation=activation_func,
            kernel_initializer=tf.keras.initializers.HeUniform(),
        )
    )
    MLP.add(BatchNormalization())
    MLP.add(Dense(32, activation=activation_func))
    MLP.add(Dropout(dropout_rate))
    MLP.add(Dense(32, activation=activation_func))
    MLP.add(Dropout(dropout_rate))
    MLP.add(Dense(3, activation="softmax"))
    MLP.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer_algo,
        metrics=[
            "accuracy",
            "Precision",
            "Recall",
            "AUC",
            tfa.metrics.F1Score(num_classes=3, average="macro"),
        ],
    )
    return MLP


def create_model_LSTM(activation_func="swish", dropout_rate=0.2, optimizer_algo="adam"):
    LSTM_model = Sequential()
    LSTM_model.add(
        LSTM(
            units=64,
            return_sequences=True,
            input_shape=(30, 30),
            activation=activation_func,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )
    )
    LSTM_model.add(BatchNormalization())
    LSTM_model.add(LSTM(units=32, return_sequences=True, activation=activation_func))
    LSTM_model.add(BatchNormalization())
    LSTM_model.add(Dropout(dropout_rate))
    LSTM_model.add(LSTM(units=32, return_sequences=False, activation=activation_func))
    LSTM_model.add(BatchNormalization())
    LSTM_model.add(Dropout(dropout_rate))
    LSTM_model.add(Dense(3, activation="softmax"))
    LSTM_model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer_algo,
        metrics=[
            "accuracy",
            "Precision",
            "Recall",
            "AUC",
            tfa.metrics.F1Score(num_classes=3, average="macro"),
        ],
    )
    return LSTM_model


def create_model_GRU(activation_func="swish", dropout_rate=0.2, optimizer_algo="adam"):
    GRU_model = Sequential()
    GRU_model.add(
        GRU(
            units=64,
            return_sequences=True,
            input_shape=(30, 30),
            activation=activation_func,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )
    )
    GRU_model.add(BatchNormalization())
    GRU_model.add(GRU(units=32, return_sequences=True, activation=activation_func))
    GRU_model.add(BatchNormalization())
    GRU_model.add(Dropout(dropout_rate))
    GRU_model.add(GRU(units=32, return_sequences=False, activation=activation_func))
    GRU_model.add(BatchNormalization())
    GRU_model.add(Dropout(dropout_rate))
    GRU_model.add(Dense(3, activation="softmax"))
    GRU_model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer_algo,
        metrics=[
            "accuracy",
            "Precision",
            "Recall",
            "AUC",
            tfa.metrics.F1Score(num_classes=3, average="macro"),
        ],
    )
    return GRU_model


def create_model_CNN_2D(
    activation_func="relu",
    dropout_rate=0.2,
    optimizer_algo="adam",
    kernel=3,
    pooling=3,
):
    CNN_2D = Sequential()
    CNN_2D.add(
        Conv2D(
            filters=64,
            kernel_size=(kernel, kernel),
            input_shape=(30, 30, 1),
            padding="same",
            activation=activation_func,
            kernel_initializer=tf.keras.initializers.HeUniform(),
        )
    )
    CNN_2D.add(BatchNormalization())
    CNN_2D.add(MaxPooling2D(pool_size=(pooling, pooling), padding="same"))
    CNN_2D.add(BatchNormalization())
    CNN_2D.add(
        Conv2D(
            filters=64,
            kernel_size=(kernel, kernel),
            padding="same",
            activation=activation_func,
        )
    )
    CNN_2D.add(BatchNormalization())
    CNN_2D.add(MaxPooling2D(pool_size=(pooling, pooling), padding="same"))
    CNN_2D.add(BatchNormalization())
    CNN_2D.add(Dropout(dropout_rate))
    CNN_2D.add(Flatten())
    CNN_2D.add(Dense(32, activation=activation_func))
    CNN_2D.add(Dropout(dropout_rate))
    CNN_2D.add(Dense(3, activation="softmax"))
    CNN_2D.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer_algo,
        metrics=[
            "accuracy",
            "Precision",
            "Recall",
            "AUC",
            tfa.metrics.F1Score(num_classes=3, average="macro"),
        ],
    )
    return CNN_2D
