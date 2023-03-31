import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

data_path = os.path.dirname(__file__)

# load
model_final = joblib.load(data_path + "/model_final.pkl")
scaler = joblib.load(data_path + "/scaler.bin")
test_samples = pd.read_csv(data_path + "/test_samples.csv", index_col=0)

# make test data and scale
X_test, y_test = test_samples.drop("cardio", axis=1), test_samples["cardio"]
X_test_scaled = scaler.transform(X_test)

prob0 = model_final.predict_proba(X_test_scaled)[:, 0]
prob1 = model_final.predict_proba(X_test_scaled)[:, 1]
pred = model_final.predict(X_test_scaled)

df = pd.DataFrame(
    {"probability class 0": prob0, "probability class 1": prob1, "prediction": pred, "actual": y_test.values}, index=test_samples.index
)

df.to_csv(data_path + "/prediction.csv")