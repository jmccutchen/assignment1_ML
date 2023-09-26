
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import numpy as np
print("Hi ")
# 'spoofing-hackrf_vehicle_attitude.csv'
df = pd.concat(
    map(pd.read_csv, ['benign-log_radio_status_3.csv', 'jamming-log_radio_status_3.csv','spoofing-hackrf-log_radio_status_3.csv']),
    ignore_index=True)

print(df.head())

print(df)

col_name_list = list(df.columns)
print("col_name_list = ", col_name_list)
size_col_name_list = len(col_name_list)
print("size_col_name_list = ", size_col_name_list)

print("df.shape ", df.shape)

X = df.drop(labels = ["attack"],axis = 1)
Y = df["attack"].values

# Create Train & Test Data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
random_state=101)

print("X_train ", X_train)
print("X_test ", X_test)
print("y_train ", y_train)
print("y_test ", y_test)

knn = KNeighborsClassifier()
knn = KNeighborsClassifier(n_neighbors=79, weights='uniform')
knn.fit(X_train, y_train)

# #Model Evaluation
y_pred_knn=knn.predict(X_test)
print("y_pred_knn ", y_pred_knn)
print("y_test ", y_test)
cm_knn=confusion_matrix(y_test, y_pred_knn)

print(cm_knn)
labels = ['Not attacked', 'attacked']
plt.figure(figsize=(7,5))
ax= plt.subplot()
sns.heatmap(cm_knn,cmap="Reds",annot=True,fmt='.1f', ax = ax);

# labels, title and ticks
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix KNN');
plt.show()

# cm_knn_nor = cm_knn.astype('float') / cm_knn.sum(axis=1)[:, np.newaxis]
# labels = ['Not attacked', 'attacked']
# plt.figure(figsize=(7,5))
# ax= plt.subplot()
# sns.heatmap(cm_knn_nor,cmap="Greens", annot=True,fmt='.1f', ax = ax);
# # labels, title and ticks
# ax.set_xticklabels(labels)
# ax.set_yticklabels(labels)
# ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
# ax.set_title('Normalized Confusion Matrix KNN');
# plt.show()
