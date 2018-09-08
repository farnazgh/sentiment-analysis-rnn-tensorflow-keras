#!/usr/bin/env python

#import
# ===========================================================
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist

from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences




# load data
# ===========================================================
from keras.datasets import imdb 



(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",num_words=None,skip_top=0,maxlen=None,seed=113,start_char=1,oov_char=2,index_from=3)


print("Train-set size: ", len(x_train))
print("Test-set size:  ", len(x_test))


data_text = x_train + x_test


# print(x_test[1])
# print(y_test[1])

# padding data
# ==========================================================
num_tokens = [len(tokens) for tokens in x_train + x_test]
num_tokens = np.array(num_tokens)
# print(np.mean(num_tokens))
# print(np.max(num_tokens))


max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
# print(max_tokens)

# print(np.sum(num_tokens < max_tokens) / len(num_tokens))

pad = 'pre'
x_train_pad = pad_sequences(x_train, maxlen=max_tokens,padding=pad, truncating=pad)
x_test_pad = pad_sequences(x_test, maxlen=max_tokens, padding=pad, truncating=pad)

# print(x_train_pad.shape)
# print(x_test_pad.shape)

# print(x_train[0])
# print(x_train_pad[0])



# model
# ===========================================================

model = Sequential()

embedding_size = 8
max =0
num_words = 89000



model.add(Embedding(input_dim=num_words,output_dim=embedding_size,input_length=max_tokens,name='layer_embedding'))
model.add(GRU(units=16, return_sequences=True))

model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(lr=1e-3)

model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

# print(model.summary())


# train the model
# =========================================================
model.fit(x_train_pad, y_train,validation_split=0.05, epochs=3, batch_size=64)


# epoch 1/3 [==============================] - 499s 21ms/step - loss: 0.4832 - acc: 0.7464 - val_loss: 0.3709 - val_acc: 0.8472
# epoch 2/3 [==============================] - 520s 22ms/step - loss: 0.2163 - acc: 0.9221 - val_loss: 0.3191 - val_acc: 0.8760
# epoch 3/3 [==============================] - 495s 21ms/step - loss: 0.1185 - acc: 0.9626 - val_loss: 0.3562 - val_acc: 0.8656


# performance on Test-set
# ==========================================================
result = model.evaluate(x_test_pad, y_test)
print("Accuracy: {0:.2%}".format(result[1]))

# 25000/25000 [==============================] - 175s 7ms/step Accuracy: 85.27%