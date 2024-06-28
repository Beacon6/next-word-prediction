import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.layers import InputLayer, LSTM, Activation, Dense  # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.optimizers import RMSprop  # type: ignore

real_data_path = "data/real.csv"
fake_data_path = "data/fake.csv"


def load_data(real_data_path: str, fake_data_path: str):
    real_data = pd.read_csv(real_data_path)
    fake_data = pd.read_csv(fake_data_path)
    combined_data = pd.concat([real_data, fake_data], ignore_index=True)
    combined_data = " ".join(list(combined_data.text.values)).lower()

    return combined_data[:100000]


def tokenize(data: str):
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(data)

    return tokens


data = load_data(real_data_path, fake_data_path)
tokens = tokenize(data)
unique_tokens = np.unique(tokens)
indexed_tokens =  {token: idx for idx, token in enumerate(unique_tokens)}

n_words = 10
input_words = []
predicted_words = []

for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    predicted_words.append(tokens[i + n_words])

X = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool)
y = np.zeros((len(predicted_words), len(unique_tokens)), dtype=bool)

for i, words in enumerate(input_words):
    for j, word in (enumerate(words)):
        X[i, j, indexed_tokens[word]] = 1
    y[i, indexed_tokens[predicted_words[i]]] = 1

model = Sequential()
model.add(InputLayer(shape=(n_words, len(unique_tokens))))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(unique_tokens)))
model.add(Activation("softmax"))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.fit(X, y, batch_size=128, epochs=30, shuffle=True)

model.save("pre_trained_model.keras")
