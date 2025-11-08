#knn algorithm

import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  

#import data

data = pd.read_csv('data.csv')
print(data.head())
print(data.info())

#split columns into x and y

x = data[['sepal_length','sepal_width','petal_length','petal_width']]
y = data['species']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state =  5)

from sklearn.preprocessing import LabelEncoder, StandardScaler

#Label encoder converts object data into numerical data

label_encoder = LabelEncoder()
label_encoder.fit_transform(y_train)
label_encoder.transform(y_test)

#Standard scaler brings the column inbetween -1 and 1

standard_scaler = StandardScaler()
standard_scaler.fit_transform(x_train)
standard_scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

#confusion matrix shows errors in data

from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, y_pred) 

sns.heatmap(matrix,annot = True, fmt = 'd')
plt.title('confusion matrix')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()
