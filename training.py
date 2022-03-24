# Standard stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Machine Learning
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
import pickle
from sklearn.preprocessing import StandardScaler

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

# Algorithmic trading
import tpqoa
from datetime import datetime, timedelta
import time

# My designs
import agents
import environments
import models
import models_pytorch
import shared_optimizers
import workers
import memory
import utils
from parallel_env import ParallelEnv

def get_scaler(env, epochs):
    states = []

    for i in range(epochs):
        done = False
        while not done:   # play as random agent to generate sample space of states
            action = np.random.choice(env.action_space)
            state, reward, done, info = env.step(action)
            states.append(state)
    
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def play_one_episode(env, agent, scaler):
    state = env.reset()
    state = scaler.transform([state])
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])
        agent.train(state, action, reward, next_state, done)
        state = next_state
    
    return info["cur_val"]

def get_data(api, instrument, bar_length, months = 6):     
    now = datetime.utcnow()
    now = now - timedelta(microseconds = now.microsecond)
    past = now - timedelta(days = 30 * months)

    df_ask = api.get_history(
        instrument = instrument,
        start = past,
        end = now,
        granularity = "S5",
        price = "A",
        localize = False
    ).c.rename("Ask")

    df_ask = df_ask.resample(pd.to_timedelta(bar_length), label = "right").last()

    df_bid = api.get_history(
        instrument = instrument,
        start = past,
        end = now,
        granularity = "S5",
        price = "B",
        localize = False
    ).c.rename("Bid")

    df_bid = df_bid.resample(pd.to_timedelta(bar_length), label = "right").last()

    df = pd.concat([df_ask, df_bid], axis = 1).dropna().iloc[ : -1]
    return df

if __name__ == "__main__":
    balance = 99872.6231

    data = pd.read_csv("eur_usd_data.csv", index_col="time")

    df = data.copy()
    df["returns"] = np.log(data["Ask"] / data["Ask"].shift())
    df["dir"] = np.where(df["returns"] > 0 , 1, 0)
    df["sma"] = df["Ask"].rolling(50).mean() - df["Ask"].rolling(150).mean()
    df["boll"] = (df["Ask"] - df["Ask"].rolling(50).mean()) / df["Ask"].rolling(50).std()
    df["min"] = df["Ask"].rolling(50).min() / df["Ask"] - 1
    df["max"] = df["Ask"].rolling(50).max() / df["Ask"] - 1
    df["mom"] = df["Ask"].rolling(3).mean()
    df["vol"] = df["Ask"].rolling(50).std()
    df.dropna(inplace = True)

    lags = 5
    cols = ["Ask", "Bid", "returns", "dir", "sma", "boll", "min", "max", "mom", "vol"]
    features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]

    for f in features:
        for lag in range(1, lags + 1):
            col = "{}_lag_{}".format(f, lag)
            df[col] = df[f].shift(lag)
            cols.append(col)
    df.dropna(inplace = True)

    state_size = len(cols) + 2

    mp.set_start_method("spawn")

    train_sim = ParallelEnv(
        input_dims = state_size,
        data = df, 
        initial_investment = balance,
        bar_length = "20min",
        time_horizon = "12hour",
        units = 100000
    )