import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  

data = pd.read_csv('data.csv')
print(data.head())
print(data.info())

x = data[['sepal_length','sepal_width','petal_length','petal_width']]
y = data['species']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state =  5)

from sklearn.preprocessing import LabelEncoder, StandardScaler

label_encoder = LabelEncoder()
label_encoder.fit_transform(y_train)
label_encoder.transform(y_test)

standard_scaler = StandardScaler()
standard_scaler.fit_transform(x_train)
standard_scaler.transform(x_test)
