from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

from Data import Data

max_features = 1024

# create model
model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(2, activation="sigmoid"))

# compile model
model.compile(loss="binary_crossentropy",
				optimizer="rmsprop",
				metrics=["accuracy"])

# get training data
in_train, out_train = Data.extract("tracking.txt", "sensor.txt")
in_test, out_test = Data.extract("tracking.txt", "sensor.txt")
#in_train, out_train, in_test, out_test = Data.generate_dummy_data()

# train model
model.fit(in_train, out_train, batch_size=2, epochs=10)

# evaluate model
#score = model.evaluate(in_test, out_test, batch_size=16)
score = model.evaluate(in_test, out_test, verbose = 0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

########
# SAVE #
########
# source: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# source: https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model

model_json = model.to_json()
# with is similar to try/ catch but better(?)
with open("model.json", "w") as json_file:
	json_file.write(model_json)

model.save_weights("model.h5")

########
# LOAD #
########

# load json and create model
json_file = open("model.json", "r")
loaded_model_json = open("model.json", "r").read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
score = loaded_model.evaluate(in_test, out_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))