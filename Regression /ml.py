import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('')
x = dataset.iloc[ : , :-1].values
y = dataset.iloc[ : , 3].values

#for missing data values

from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values = 'NaN' , strategy = 'mean' , axis =0 )
imputer = Imputer.fit(x[:, 1:3])
x[:,1:3] = Imputer.transform(x[:,1:3])

#for encosing cateogorical data 

from sklearn.preprocessing import OneHotEncoder , LabelEncoder
labelencoder = LabelEncoder()
x[:,0] = labelencoder.fit_transform(x[: 0])
onehotencoder= OneHotEncoder(cateogorical_feaurures = [0])
x = onehotencoder.fit_transform(x).toarray()
labelencoder1= LabelEncoder()
y = labelencoder1.fit_transform(y)

#Splitting into test train data 

from sklearn.cross_validation import test_train_split
x_train , x_test , y_train , y_test = test_train_split()

#for noramlization and standaridization 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc.fit_transform(x)
x_test = sc.transform(x_test)

