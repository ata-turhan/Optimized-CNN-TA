import time
import copy
import os

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
from .configurations import set_random_seed
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
import optuna


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

def model_train_test(model_name, datas, epochs=100, parameters=None, seed=42):
    set_random_seed(seed)

    start_time = time.time()
    predictions = []
    f1_scores = []

    model_creations = {
        "MLP": create_model_MLP,
        "LSTM": create_model_LSTM,
        "GRU": create_model_GRU,
        "CNN_2D": create_model_CNN_2D,
    }
    create_model = model_creations[model_name]
    OUTPUT_PATH = "./outputs"
    for i in range(len(datas)):
        es = EarlyStopping(
            monitor="val_f1_score", mode="max", verbose=0, patience=30, min_delta=1e-3
        )
        mcp = ModelCheckpoint(
            os.path.join(OUTPUT_PATH, f"best_{model_name}_model-{i+1}.h5"),
            monitor="val_f1_score",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="max",
        )

        if model_name == "MLP":
            val_split_point = int(0.8 * len(datas[i][0]))
            X_train = datas[i][0][29:val_split_point].iloc[:, :-1]
            y_train = tf.keras.utils.to_categorical(
                datas[i][0][29:val_split_point].iloc[:, -1], num_classes=3
            )
            X_val = datas[i][0][val_split_point:].iloc[:, :-1]
            y_val = tf.keras.utils.to_categorical(
                datas[i][0][val_split_point:].iloc[:, -1], num_classes=3
            )
            X_test = datas[i][1][29:].iloc[:, :-1]
            y_test = datas[i][1][29:].iloc[:, -1]
        else:
            val_split_point = int(0.8 * len(datas[i][0]))
            X_train = datas[i][0][:val_split_point]
            y_train = datas[i][1][:val_split_point]
            X_val = datas[i][0][val_split_point:]
            y_val = datas[i][1][val_split_point:]
            X_test = datas[i][2]
            y_test = datas[i][3]

        if parameters is None:
            model = create_model()
            batch_size = 32
        else:
            model = create_model(
                parameters["activation_func"],
                parameters["dropout_rate"],
                parameters["optimizer_algo"],
            )
            batch_size = parameters["batch_size"]
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            callbacks=[es, mcp],
            validation_data=(X_val, y_val),
            class_weight={0: 1, 1: 10, 2: 10},
        )
        y_pred = model.predict(X_test)
        y_pred = y_pred.argmax(axis=-1)
        predictions.append(y_pred)
        f1_scores.append(f1_score(y_test, y_pred, average="macro"))
    minutes = round(int(time.time() - start_time) / 60, 2)
    return (predictions, np.mean(f1_scores), minutes)


def model_ho(model_name, datas, epochs=30, parameter_space:dict={}, seed=42, trial_number=5):
    set_random_seed(seed)
    start_time = time.time()

    def objective(trial):
        activation_func = trial.suggest_categorical(
            name="activation_func", choices=parameter_space["activation_func"]
        )
        dropout_rate = trial.suggest_categorical(
            "dropout_rate", parameter_space["dropout_rate"]
        )
        optimizer_algo = trial.suggest_categorical(
            "optimizer_algo", parameter_space["optimizer_algo"]
        )
        batch_size = trial.suggest_categorical(
            "batch_size", parameter_space["batch_size"]
        )
        # lr_max = trial.suggest_categorical("learning_rate_max", [1e-1, 1e-2, 1e-3, 1e-4])
        parameters = {
            "activation_func": activation_func,
            "dropout_rate": dropout_rate,
            "optimizer_algo": optimizer_algo,
            "batch_size": batch_size,
        }
        ho_datas = copy.deepcopy(datas)
        for i in range(len(datas)):
            val_split_point = int(0.7 * len(datas[i][0]))
            if model_name == "MLP":
                ho_datas[i][1] = datas[i][0][val_split_point:]
                ho_datas[i][0] = datas[i][0][:val_split_point]
            else:
                ho_datas[i][2] = datas[i][0][val_split_point:]
                ho_datas[i][0] = datas[i][0][:val_split_point]
                ho_datas[i][3] = datas[i][1][val_split_point:].argmax(axis=-1)
                ho_datas[i][1] = datas[i][1][:val_split_point]
        return model_train_test(model_name, ho_datas, epochs=epochs, parameters=parameters, seed=seed)[1]

    study = optuna.create_study(
        study_name=f"{model_name}_Bayesian_Optimization",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=trial_number)
    trial = study.best_trial

    print("\n------------------------------------------")
    print("Best F1 Macro: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
    minutes = round(int(time.time() - start_time) / 60, 2)
    print(f"\nCompleted in {minutes} minutes")
    return trial.params
