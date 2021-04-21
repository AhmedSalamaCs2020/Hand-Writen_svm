# import numpy as np
# import pandas as pd
# from utiles.constants import Constants


# FUNCTIONS
# def predict_outcome(feature_matrix, weights):
#    return np.dot(feature_matrix, weights)


# kc_house_data = pd.read_csv(Constants.kc_house_data)
# kc_house_train_data = pd.read_csv(Constants.kc_house_train_data)
# kc_house_test_data = pd.read_csv(Constants.kc_house_test_data)

# HOLD DATA
# list_features = ["constant","sqft_living", "bedrooms","price"]
# df_1 = pd.DataFrame(kc_house_data, columns=list_features)

# df_1["constant"] = 1

# print (df_1)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

digits = load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# fit the model with data
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)
from sklearn import svm
from sklearn.metrics import accuracy_score

svm_classifier = svm.SVC(gamma=0.001)
svm_classifier.fit(X_train, y_train)
y_hat = svm_classifier.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_test, y_hat)
print(accuracy)
