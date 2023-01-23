#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import yfinance as yf
from matplotlib import pyplot as plt
from pylab import rcParams
import sklearn
from sklearn.linear_model import LinearRegression
import talib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MaxAbsScaler
import time
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.metrics import *
import os
import tensorflow as tf
import tensorflow_addons as tfa 
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
import optuna


# In[2]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# <h1 style="font-size:40px;"> <center> FUNCTIONS </center> </h1>

# In[25]:


SEED = 42


def set_random_seed():
    tf.keras.utils.set_random_seed(
    SEED
)
    

def HMA(df:pd.DataFrame, timeperiod:int= 14) -> float:
    """
    Hull Moving Average.
    Formula:
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    hma = talib.WMA(2 * talib.WMA(df, int(timeperiod/2)) - talib.WMA(df, timeperiod), int(np.sqrt(timeperiod)))
    return hma


def money_flow_volume_series(df: pd.DataFrame) -> pd.Series:
    """
    Calculates money flow series
    """
    mfv = df['Volume'] * (2*df['Close'] - df['High'] - df['Low']) / \
                                    (df['High'] - df['Low'])
    return mfv


def money_flow_volume(df: pd.DataFrame, timeperiod: int=20) -> pd.Series:
    """
    Calculates money flow volume, or q_t in our formula
    """
    return money_flow_volume_series(df).rolling(timeperiod).sum()


def CMF(df: pd.DataFrame, timeperiod: int=20) -> pd.Series:
    """
    Calculates the Chaikin money flow
    """
    return money_flow_volume(df, timeperiod) / df['Volume'].rolling(timeperiod).sum()


def pltcolor(lst:list) -> list:
    cols=[]
    for i in range(lst.shape[0]):
        if lst.iloc[i] == 1:
            cols.append('green')
        elif lst.iloc[i] == 2:
            cols.append('red')
    return cols
     

def trendNormalizePrices(prices:pd.DataFrame) -> None:
    df = prices.copy()
    df["rowNumber"] = list(range(len(df)))
    df["TN_Open"] = list(range(len(df)))
    df["TN_High"] = list(range(len(df)))
    df["TN_Low"] = list(range(len(df)))
    df["TN_Close"] = list(range(len(df)))
    for i in range(29,len(df)):
        model = LinearRegression()
        model.fit(np.array(df["rowNumber"].iloc[i-29:i+1]).reshape(-1,1), np.array(df["Close"].iloc[i-29:i+1]))
        prediction = model.predict(np.array([df["rowNumber"].iloc[i]]).reshape(-1,1))
        df.iloc[i, df.columns.get_loc("TN_Open")] = df["Open"].iloc[i] - prediction 
        df.iloc[i, df.columns.get_loc("TN_High")] = df["High"].iloc[i] - prediction 
        df.iloc[i, df.columns.get_loc("TN_Low")] = df["Low"].iloc[i] - prediction 
        df.iloc[i, df.columns.get_loc("TN_Close")] = df["Close"].iloc[i] - prediction 
    df["Open"] = df["TN_Open"] 
    df["High"] = df["TN_High"]
    df["Low"] = df["TN_Low"]
    df["Close"] = df["TN_Close"]
    df = df.drop(index=df.index[:30], axis=0)
    df = df.drop(columns=["TN_Open", "TN_High", "TN_Low", "TN_Close", "rowNumber"], axis=1)
    return df
    
    
def create_labels(prices:pd.DataFrame) -> None:
    df = prices.copy()
    df["Label"] = [0] * df.shape[0]
    for i in range(df.shape[0]-10):
        s = set(df["Close"].iloc[i:i+11]) 
        minPrice = sorted(s)[0]
        maxPrice = sorted(s)[-1]
        for j in range(i, i+11):
            if df["Close"].iloc[j] == minPrice and (j-i) == 5:
                df.iloc[j, df.columns.get_loc('Label')] = 1
            elif df["Close"].iloc[j] == maxPrice and (j-i) == 5:
                df.iloc[j, df.columns.get_loc('Label')] = 2
    return df.iloc[6:-6]
                
                
def reverse_one_hot(predictions:np.array) -> np.array:
    return np.argmax(predictions, axis=1)


def one_hot(predictions:np.array) -> np.array:
    predictions_one_hot = []
    for i in predictions:
        prediction = [0,0,0]
        prediction[int(i)] = 1
        predictions_one_hot.append(prediction)   
    return np.array(predictions_one_hot)


def number_null_and_nan(df:pd.DataFrame) -> int:
    na = pd.isna(df).sum().sum()
    null = df.isnull().sum().sum()
    return (na+null) 


# <h1 style="font-size:40px;"> <center> DATA PREPROCESSING </center> </h1>

# In[4]:


prices = yf.download("SPY", start="2009-09-20", end="2023-01-01", interval="1d", progress=False, auto_adjust=True)
prices


# In[ ]:


plt.figure(figsize=(20,10))
plt.title("SPY Price 2010-2022")
plt.xlabel("Date")
plt.ylabel("Price")
plt.plot(prices[["Close"]].iloc[150:,:])


# <h1 style="font-size:30px;"> <center> Create Labels and Visualize </center> </h1>

# In[11]:


trendNormalizePrices(prices_with_label)


# <h1 style="font-size:30px;"> <center> Adding Technical Indicators </center> </h1>

# In[5]:


prices_and_indicators = prices.copy()


# In[6]:


for i in range(7,30):
    prices_and_indicators[f"RSI-{i}"] = talib.RSI(prices_and_indicators["Close"], timeperiod = i)
    prices_and_indicators[f"WILLR-{i}"] = talib.WILLR(prices_and_indicators["High"],prices_and_indicators["Low"],prices_and_indicators["Close"], timeperiod = i)
    prices_and_indicators[f"STOCH-{i}"] = talib.STOCH(prices_and_indicators["High"],prices_and_indicators["Low"],prices_and_indicators["Close"], fastk_period=i+7, slowk_period=i-4)[0]
    prices_and_indicators[f"STOCHF-{i}"] = talib.STOCHF(prices_and_indicators["High"],prices_and_indicators["Low"],prices_and_indicators["Close"], fastk_period=i-2, fastd_period=i-4)[0]
    prices_and_indicators[f"SMA-{i}"] = talib.SMA(prices_and_indicators["Close"], timeperiod = i)
    prices_and_indicators[f"EMA-{i}"] = talib.EMA(prices_and_indicators["Close"], timeperiod = i)
    prices_and_indicators[f"WMA-{i}"] = talib.WMA(prices_and_indicators["Close"], timeperiod = i)
    prices_and_indicators[f"HMA-{i}"] = HMA(prices_and_indicators["Close"], timeperiod = i)
    prices_and_indicators[f"TEMA-{i}"] = talib.TEMA(prices_and_indicators["Close"], timeperiod = i)
    prices_and_indicators[f"PPO-{i}"] = talib.PPO(prices_and_indicators["Close"], fastperiod=i, slowperiod=i+14)
    prices_and_indicators[f"ROC-{i}"] = talib.ROC(prices_and_indicators["Close"], timeperiod = i)
    prices_and_indicators[f"CMO-{i}"] = talib.CMO(prices_and_indicators["Close"], timeperiod = i)
    prices_and_indicators[f"MACD-{i}"] = talib.MACD(prices_and_indicators["Close"], fastperiod=i, slowperiod=i+14)[0]
    prices_and_indicators[f"MAMA-{i}"] = talib.MAMA(prices_and_indicators["Close"], fastlimit=1/i, slowlimit=1/(i+14))[0]
    prices_and_indicators[f"STOCHRSI-{i}"] = talib.STOCHRSI(prices_and_indicators["Close"], timeperiod=i)[0]
    prices_and_indicators[f"DX-{i}"] = talib.DX(prices_and_indicators["High"],prices_and_indicators["Low"],prices_and_indicators["Close"], timeperiod = i)
    prices_and_indicators[f"ADXR-{i}"] = talib.ADXR(prices_and_indicators["High"],prices_and_indicators["Low"],prices_and_indicators["Close"], timeperiod = i)
    prices_and_indicators[f"CCI-{i}"] = talib.CCI(prices_and_indicators["High"],prices_and_indicators["Low"],prices_and_indicators["Close"], timeperiod = i)
    prices_and_indicators[f"PLUS_DI-{i}"] = talib.PLUS_DI(prices_and_indicators["High"],prices_and_indicators["Low"],prices_and_indicators["Close"], timeperiod = i)
    prices_and_indicators[f"MINUS_DI-{i}"] = talib.MINUS_DI(prices_and_indicators["High"],prices_and_indicators["Low"],prices_and_indicators["Close"], timeperiod = i)
    prices_and_indicators[f"ATR-{i}"] = talib.ATR(prices_and_indicators["High"],prices_and_indicators["Low"],prices_and_indicators["Close"], timeperiod = i)
    prices_and_indicators[f"SAR-{i}"] = talib.SAR(prices_and_indicators["High"],prices_and_indicators["Low"], maximum = 1/i)
    prices_and_indicators[f"PLUS_DM-{i}"] = talib.PLUS_DM(prices_and_indicators["High"],prices_and_indicators["Low"], timeperiod = i)
    prices_and_indicators[f"AROONOSC-{i}"] = talib.AROONOSC(prices_and_indicators["High"],prices_and_indicators["Low"], timeperiod = i)
    prices_and_indicators[f"MIDPRICE-{i}"] = talib.MIDPRICE(prices_and_indicators["High"],prices_and_indicators["Low"], timeperiod = i)
    prices_and_indicators[f"MFI-{i}"] = talib.MFI(prices_and_indicators["High"],prices_and_indicators["Low"],prices_and_indicators["Close"],prices_and_indicators["Volume"], timeperiod = i)
    prices_and_indicators[f"ADOSC-{i}"] = talib.ADOSC(prices_and_indicators["High"],prices_and_indicators["Low"],prices_and_indicators["Close"],prices_and_indicators["Volume"], fastperiod=i-4, slowperiod=i+3)
    prices_and_indicators[f"BBANDS-{i}"] = talib.BBANDS(prices_and_indicators["Close"], timeperiod = i)[1]
    prices_and_indicators[f"CMF-{i}"] = CMF(prices_and_indicators, timeperiod = i)
prices_and_indicators["BOP"] = talib.BOP(prices_and_indicators["Open"],prices_and_indicators["High"],prices_and_indicators["Low"],prices_and_indicators["Close"])
prices_and_indicators["TRANGE"] = talib.TRANGE(prices_and_indicators["High"],prices_and_indicators["Low"],prices_and_indicators["Close"])    
prices_and_indicators["SAREXT"] = talib.SAREXT(prices_and_indicators["High"],prices_and_indicators["Low"])
prices_and_indicators["AD"] = talib.AD(prices_and_indicators["High"],prices_and_indicators["Low"],prices_and_indicators["Close"],prices_and_indicators["Volume"])
prices_and_indicators["OBV"] = talib.OBV(prices_and_indicators["Close"],prices_and_indicators["Volume"])
prices_and_indicators.dropna(inplace = True)


# In[7]:


prices_and_indicators


# <h1 style="font-size:30px;"> <center> Data Labeling </center> </h1>

# In[8]:


prices_and_indicators_with_label = create_labels(prices_and_indicators)
prices_and_indicators_with_label


# In[9]:


prices_and_indicators_with_label["Label"].value_counts()


# In[10]:


rcParams['figure.figsize'] = 20, 10
plt.figure(figsize=(50, 30))
prices_and_indicators_with_label[["Close"]].plot(kind="line", stacked=False,linewidth=1)
buy_and_sell_preds = prices_and_indicators_with_label.query('Label != 0')
plt.scatter(x = buy_and_sell_preds.index, y = buy_and_sell_preds.Close, s=5, c=pltcolor(buy_and_sell_preds.Label))
plt.show() 


# <h1 style="font-size:30px;"> <center> Creating Train & Test Data </center> </h1>

# In[11]:


prices_and_indicators_with_label.info()


# In[12]:


datas = []

for i in range(5, 13):
    train = prices_and_indicators_with_label.loc[ (prices_and_indicators_with_label.index >= f"{2010+i-5}") & (prices_and_indicators_with_label.index <= f"{2010+i}") ]
    test = prices_and_indicators_with_label.loc[ (prices_and_indicators_with_label.index >= f"{2010+i}") & (prices_and_indicators_with_label.index <= f"{2010+i+1}") ]
    datas.append([train, test])


# <h1 style="font-size:30px;"> <center> Feature Selection </center> </h1>

# In[13]:


for i in range(len(datas)):
    selected_feature_count = 30
    select = SelectKBest(score_func=f_classif, k = selected_feature_count)
    fitted = select.fit(datas[i][0].iloc[:,:-1], datas[i][0].iloc[:,-1])
    train_features = fitted.transform(datas[i][0].iloc[:,:-1])
    test_features = fitted.transform(datas[i][1].iloc[:,:-1])
    
    selected_features_boolean = select.get_support()
    features = list(datas[i][1].columns[:-1])
    selected_features = []
    for j in range(len(features)):
        if selected_features_boolean[j]:
            selected_features.append(features[j])
    train_label = datas[i][0].Label
    test_label = datas[i][1].Label
    
    datas[i][0] = pd.DataFrame(data=train_features.astype('float32'), columns=selected_features, index=datas[i][0].index)
    datas[i][0]["Label"] = train_label
    datas[i][1] = pd.DataFrame(data=test_features.astype('float32'), columns=selected_features, index=datas[i][1].index)
    datas[i][1]["Label"] = test_label


# In[14]:


datas[0][0]


# In[15]:


datas[0][1]


# In[16]:


for i in range(len(datas)):
    abs_scaler = MaxAbsScaler()
    abs_scaler.fit(datas[i][0])
    scaled_train = abs_scaler.transform(datas[i][0])
    scaled_test = abs_scaler.transform(datas[i][1])
    datas[i][0] = pd.DataFrame(data=scaled_train, columns=datas[i][0].columns, index=datas[i][0].index)
    datas[i][0]["Label"] = datas[i][0]["Label"] * 2
    datas[i][1] = pd.DataFrame(data=scaled_test, columns=datas[i][1].columns, index=datas[i][1].index)
    datas[i][1]["Label"] = datas[i][1]["Label"] * 2


# In[17]:


datas[0][0]


# In[18]:


datas[0][1]


# <h1 style="font-size:30px;"> <center> Controling Null Values </center> </h1>

# In[19]:


total_na_count = 0
for data in datas:
    total_na_count += number_null_and_nan(data[0])
    total_na_count += number_null_and_nan(data[1])
print(f"Total null and nan values = {total_na_count}")


# <h1 style="font-size:40px;"> <center> MODEL INITIALIZATIONS </center> </h1>

# <h1 style="font-size:30px;"> <center> MLP </center> </h1>

# In[20]:


def create_model_MLP(trial=None, activation_func="swish", dropout_rate = 0.2, optimizer_algo = "adam"):
    MLP = Sequential()
    MLP.add(Dense(64, input_shape=(30,), activation=activation_func, kernel_initializer=tf.keras.initializers.HeUniform()))
    MLP.add(BatchNormalization())
    MLP.add(Dense(32, activation=activation_func))
    MLP.add(Dropout(dropout_rate))
    MLP.add(Dense(32, activation=activation_func))
    MLP.add(Dropout(dropout_rate))
    MLP.add(Dense(3, activation='softmax'))
    MLP.compile(loss="categorical_crossentropy", optimizer=optimizer_algo, metrics=["accuracy","Precision","Recall","AUC",tfa.metrics.F1Score(num_classes=3, average="macro")])
    return MLP


# In[22]:


set_random_seed()

start_time = time.time()
predictions = []
f1_scores = []

for i in range(len(datas)):
    OUTPUT_PATH = "./outputs"
    es = EarlyStopping(monitor='val_f1_score', mode='max', verbose=1, patience=20, min_delta=1e-2)
    mcp = ModelCheckpoint(os.path.join(OUTPUT_PATH,f"best_CNN_model-{i+1}.h5"), monitor='val_f1_score', verbose=0, 
                          save_best_only=True, save_weights_only=False, mode='max')
    
    val_split_point = int(0.5*len(datas[i][0]))
    X_train = datas[i][0][:val_split_point].iloc[:, :-1]
    y_train = tf.keras.utils.to_categorical(datas[i][0][:val_split_point].iloc[:, -1], num_classes = 3)
    X_val = datas[i][0][val_split_point:].iloc[:, :-1]
    y_val = tf.keras.utils.to_categorical(datas[i][0][val_split_point:].iloc[:, -1], num_classes = 3)
    X_test = datas[i][1].iloc[:, :-1]
    y_test = datas[i][1].iloc[:, -1]
    
    model = create_model_MLP()
    model.fit(X_train, y_train, batch_size=64, 
                        epochs=1, verbose=0, callbacks=[es, mcp], 
                        validation_data=(X_val, y_val), 
                        class_weight={0:1, 1:10, 2:10})
    y_pred = model.predict(X_test)
    y_pred = y_pred.argmax(axis=-1)
    predictions.append(y_pred)
    f1_scores.append(f1_score(y_test, y_pred, average='macro'))
    
print(f"\nAverage f1-macro score: {np.mean(f1_scores)}\n")
minutes = round(int(time.time() - start_time)/60, 2)
print(f"\nCompleted in {minutes} minutes\n")


# <h1 style="font-size:30px;"> <center> LSTM </center> </h1>

# In[ ]:





# <h1 style="font-size:30px;"> <center> GRU </center> </h1>

# In[ ]:





# <h1 style="font-size:30px;"> <center> CNN </center> </h1>

# In[ ]:





# <h1 style="font-size:40px;"> <center> HYPERPARAMETER TUNING </center> </h1>

# <h1 style="font-size:30px;"> <center> MLP </center> </h1>

# In[28]:


set_random_seed()
start_time = time.time()

def objective(trial):
    activation_func = trial.suggest_categorical(name="activation_func", choices = ["relu", "selu", "swish"])
    dropout_rate = trial.suggest_categorical("drop_out_rate", [0.1, 0.2, 0.3])
    optimizer_algo = trial.suggest_categorical("optimizer_algorithm", ["adam", "adadelta", "rmsprop"])
    batch = trial.suggest_categorical("batch_size", [32, 64, 256])
    #epoch_num = trial.suggest_categorical("epoch_number", [50, 100, 200])
    lr_max = trial.suggest_categorical("learning_rate_max", [1e-1,1e-2,1e-3,1e-4])

    model = create_model_MLP(trial, activation_func, dropout_rate, optimizer_algo)

    f1_scores = []

    for i in range(len(datas)):
        OUTPUT_PATH = "./outputs"
        es = EarlyStopping(monitor='f1_score', mode='max', verbose=1, patience=20, min_delta=1e-2)
        mcp = ModelCheckpoint(os.path.join(OUTPUT_PATH,f"best_CNN_model-{i+1}.h5"), monitor='f1_score', verbose=0, 
                                  save_best_only=True, save_weights_only=False, mode='max')

        val_split_point = int(0.5*len(datas[i][0]))
        X_train = datas[i][0][:val_split_point].iloc[:, :-1]
        y_train = tf.keras.utils.to_categorical(datas[i][0][:val_split_point].iloc[:, -1], num_classes = 3)
        X_val = datas[i][0][val_split_point:].iloc[:, :-1]
        y_val = datas[i][0][val_split_point:].iloc[:, -1]

        model.fit(X_train, y_train, batch_size=batch, 
                                epochs=1, verbose=0, callbacks=[es, mcp], 
                                class_weight={0:1, 1:10, 2:10})
        y_pred = model.predict(X_val)
        y_pred = y_pred.argmax(axis=-1)
        f1_scores.append(f1_score(y_val, y_pred, average='macro'))
    return np.mean(f1_scores)

study = optuna.create_study(study_name="MLP_Bayesian_Optimization", direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=5)
trial = study.best_trial

print("\n------------------------------------------")
print('Best F1 Macro: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
minutes = round(int(time.time() - start_time)/60, 2)
print(f"\nCompleted in {minutes} minutes")


# In[30]:






# <h1 style="font-size:30px;"> <center> LSTM </center> </h1>

# In[ ]:





# <h1 style="font-size:30px;"> <center> GRU </center> </h1>

# In[ ]:





# <h1 style="font-size:30px;"> <center> CNN </center> </h1>

# In[ ]:





# In[ ]:





# In[ ]:





# <h1 style="font-size:40px;"> <center> FINANCIAL EVALUATION </center> </h1>

# <h1 style="font-size:30px;"> <center> MLP </center> </h1>

# In[ ]:





# <h1 style="font-size:30px;"> <center> LSTM </center> </h1>

# In[ ]:





# <h1 style="font-size:30px;"> <center> GRU </center> </h1>

# In[ ]:





# In[ ]:





# <h1 style="font-size:30px;"> <center> CNN </center> </h1>

# In[ ]:





# In[ ]:




