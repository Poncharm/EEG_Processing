from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

with open('ones_data.pkl', 'rb') as f:
    ones_data, ones_classes = pickle.load(f)
with open('zeros_data.pkl', 'rb') as f:
    zeros_data, zeros_classes = pickle.load(f)

# Объединение признаков и классов
features = np.concatenate((ones_data, zeros_data), axis=0)
classes = np.concatenate((ones_classes, zeros_classes), axis=0)

# Нормализация данных
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

clf = svm.SVC()
clf.fit(features_scaled, classes)

with open('svm_model.pkl', 'wb') as f:
    pickle.dump((clf, scaler), f)