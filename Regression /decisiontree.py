ort numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('/Users/savita/desktop/ML_proj/')
x = dataset.iloc[: , 1:2].values
y = dataset.iloc[ : ,2].values

from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
sc_y =  StandardScaler()
x= sc_x.fit_transform(x)
y= sc_y.fit_transform(y)



from sklearn.svm import SVR
Regressor = SVR(kernel = 'rbf')
Regressor.fit(x,y)

#inverse the transform otherwise normalized results will be obtained 

y_pred = sc_y.inverse_transform(Regressor.predict(sc_x.transform(np.array([[6.5]]))))

#visulazie the results 

plt.scatter( x , y , color='red' )
plt.plot(x , Regressor.predict(x) , color='blue' )
plt.title('Truth or bluff:SVR')
plt.xlabel('Position Levels ')
plt.ylabel('Salary')
plt.show()
