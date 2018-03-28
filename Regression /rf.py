import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('/Users/savita/desktop/AllaboutMachineLearning/Regression/Position_Salaries.csv')
x = dataset.iloc[: , 1:2].values
y = dataset.iloc[ : ,2].values

"""from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
sc_y =  StandardScaler()
x= sc_x.fit_transform(x)
y= sc_y.fit_transform(y)"""

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100 , raandom_state=0)
regressor.fit(x,y)
 
y_pred = regressor.predict(6.5)

t.scatter( x , y , color='red' )
plt.plot(x , Regressor.predict(x) , color='blue' )
plt.title('Truth or bluff:SVR')
plt.xlabel('Position Levels ')
plt.ylabel('Salary')
plt.show()


