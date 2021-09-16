import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression

from config import Config

#creating model path if it doesnt exist already
Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)



x_train = pd.read_csv(str(Config.FEATURES_PATH / "train_features.csv"))
y_train = pd.read_csv(str(Config.FEATURES_PATH /"train_labels.csv"))


model = LinearRegression()
model = model.fit(x_train, y_train.to_numpy())

#to store the model
pickle.dump(model, open(str(Config.MODELS_PATH / "model.pickle" ), "wb"))