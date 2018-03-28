import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

dataset = pd.read_csv('/Users/savita/desktop/ML_proj/Simple_Linear_Regression/Salary_Data.csv')

x= dataset.iloc[ : , 0].values
y= dataset.iloc[ : , 1].values

from sklearn.cross_validation import train_test_split
x_train , y _train , x_test , y_test = train_test_split(x , y , test_size=0.2 , random_state= 0)

from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(x_train , y_train )

y_pred = regressor.predict(x_test)

plt.scatter(x_train , y_train , color='red')
plt.plot(x_test , regressor.predict(x_train) , color= 'blue')
plt.title('salary vs experience (x_training)')
plt.xlabel(''experience)
plt.ylabel('salary')
plt.show(x)










