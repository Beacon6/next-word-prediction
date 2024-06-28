import pickle
import random
from typing import List

import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.layers import LSTM, Activation, Dense  # type: ignore
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.optimizers import RMSprop  # type: ignore

real_data_path = "data/real.csv"
fake_data_path = "data/fake.csv"


def load_data(real_data_path: str, fake_data_path: str):
    real_data = pd.read_csv(real_data_path)
    fake_data = pd.read_csv(fake_data_path)
    combined_data = pd.concat([real_data, fake_data], ignore_index=True)
    combined_data = " ".join(list(combined_data.text.values)).lower()

    return combined_data


def tokenize(data: str):
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(data)

    return tokens


data = load_data(real_data_path, fake_data_path)
tokens = tokenize(data)
