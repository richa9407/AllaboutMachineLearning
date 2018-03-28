import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('/Users/savita/desktop/ML_proj')
x = dataset.iloc[: , :-1].values
y = dataset.iloc[ : , 4].values

#Encoding the cateogorical data and Indepent Variables 

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
x[: ,3]= LabelEncoder.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#Avoiding the Dummy Variable Trap 
x=x[:, 1:]

#Train_test split
from sklearn.cross_validation import train_test_split 
x_train , y_train , x_test , y_test = train_test_split( x , y , test_size =0.2 , random_state= 0)

#fit to the trainig model

from sklearn.Linear_model import LinearRegression 
Regressor = LinearRegression()
Regressor.fit(x_train , y_train)
#test the trainig model

y_pred = Regressor.predict(x_test)

import statsmodels.formula.api as sm 
x = np.append(arr= np.ones(50,1).astype(int) , values= x , axis=1)
x_opt = x[:,[0,1,2,3,4,5]]
regressor1= sm.OLS(endog= y, exog= x_opt).fit()
regressor1.summary()

x_opt = x[:,[0,1,3,4,5]]
regressor1= sm.OLS(endog= y, exog= x_opt).fit()
regressor1.summary()


x_opt = x[:,[0,3,4,5]]
regressor1= sm.OLS(endog= y, exog= x_opt).fit()
regressor1.summary()

x_opt = x[:,[0,3,5]]
regressor1= sm.OLS(endog= y, exog= x_opt).fit()
regressor1.summary()

x_opt = x[:,[0,3,5]]
regressor1= sm.OLS(endog= y, exog= x_opt).fit()
regressor1.summary()

x_opt = x[:,[0,3]]
regressor1= sm.OLS(endog= y, exog= x_opt).fit()
regressor1.summary()




