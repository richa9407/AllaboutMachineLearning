import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('/Users/savita/desktop/ML_proj/')
x = dataset.iloc[: , 1:2].values
y = dataset.iloc[ : ,2].values


from sklearn.Linear_model import LinearRegression 
Regressor = LinearRegression()
Regressor.fit(x , y)

from sklearn.preprocessing import PolynomialFeatures
Regressor1= PolynomialFeatures( degree=3)
x_poly = Regressor1.fit_transform(x)

Regressor2 = LinearRegression()
Regressor2.fit(x_poly , y)

plt.scatter( x , y , color='red' )
plt.plot(x , Regressor.predict(x) , color='blue' )
plt.title('Truth or bluff:Linear Regression ')
plt.xlabel('Position Levels ')
plt.ylabel('Salary')
plt.show()

np.arrange(min(x), max(x) ,0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter( x , y , color='red' )
plt.plot(x_grid , Regressor2.predict(Regressor1.fit_transform(x_grid)) , color='blue' )
plt.title('Truth or bluff:Polynomial Regression ')
plt.xlabel('Position Levels ')
plt.ylabel('Salary')
plt.show()


#predicting a new result with linear regression 
Regressor.predict(6.5)

#predicting a new result with polynomial regression 
Regressor2.predict(Regressor1.fit_transform(6.5)) 

