import json
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error
import math

from config import Config

x_test = pd.read_csv(str(Config.FEATURES_PATH / 'test_features.csv'))
y_test = pd.read_csv(str(Config.FEATURES_PATH / 'test_labels.csv'))


model = pickle.load(open(str(Config.MODELS_PATH / 'model.pickle'), 'rb'))

r_squared = model.score(x_test, y_test)

y_pred = model.predict(x_test)

rmse = math.sqrt(mean_squared_error(y_test, y_pred))

with open(str(Config.METRICS_FILE_PATH), "w") as outfile:
    json.dump(dict(r_squared = r_squared, rmse=rmse), outfile)