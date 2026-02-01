import numpy as np
import pandas as pd
from pygam import LinearGAM, s, f

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

find_weekday = pd.to_datetime(data[['year', 'month', 'day']])
data['weekday'] = find_weekday.dt.dayofweek
X_train = data[['hour', 'weekday']].values
y_train = data['trips']

#s(0) spline for hour
#f(1) factor for day of week
model = LinearGAM(s(0, n_splines=24) + f(1))

modelFit = model.gridsearch(X_train, y_train)

test_data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")
find_weekday_test = pd.to_datetime(test_data[['year', 'month', 'day']])
test_data['weekday'] = find_weekday_test.dt.dayofweek
X_test = test_data[['hour', 'weekday']]
pred = modelFit.predict(X_test)
pred = np.array(pred)
