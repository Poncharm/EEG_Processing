import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('svm_model.pkl', 'rb') as f:
    clf, scaler = pickle.load(f)
with open('test_data.pkl', 'rb') as f:
    test_data, time = pickle.load(f)

T = np.linspace(time[0], time[1], len(test_data))

# Нормализация тестовых данных
test_data_scaled = scaler.transform(test_data)

# Классификация данных
prediction = clf.predict(test_data_scaled)

# Построение графика
plt.step(T, prediction)
plt.xlabel('Time')
plt.ylabel('Prediction')
plt.title('SVM Prediction over Time')
plt.show()

