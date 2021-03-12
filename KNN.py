import numpy as np
import matplotlib.pylab as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# attributes = ['BIO_ECG', 'fatiguelevel']
dataset = pd.read_csv('data/行人.csv', names=None)
print("Building dataset...")
# print(dataset.head())
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values

print("Done.")

print("Separate dataset to trainset and testset...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print("Training...")
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)

# predict test sample
y_pred = classifier.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 对给定输入进行预测
sample = pd.read_csv('data/SR_pedestrian_0.csv', names=None)
x_sample = scaler.transform(sample.iloc[:, :].values)
y_pred_sample = classifier.predict(x_sample)
print("__________")
print(y_pred_sample)
print("__________")


print("This code runing using: {}s.".format(time.process_time()))
