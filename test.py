from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

max_features = 1024

# create model
model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# generate dummy data
import numpy as np
x_train = []    # input
y_train = []    # target output
for i in range(0, 1000):
    x_train.append(np.random.random())
    y_train.append(np.sin(x_train[i]))

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = []
y_test = []
for i in range(0, 100):
    x_test.append(np.random.random())
    y_test.append(np.sin(x_test[i]))

# train model
model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)

print(score)