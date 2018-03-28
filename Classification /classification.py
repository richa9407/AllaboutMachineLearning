import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset = pd.read_csv('')
x = dataset.iloc[ : , [2,3]].values
y = dataset.iloc[: , 4].values

from sklearn.cross_validation import train_test_split 
x_train , x_test , y_train , y_test = train_test_split( x, y , test_size= 0.25 , random_state=0 )


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train= sc.fit_transform(x_train)
x_test = sc. fit_transform(x_test)

#classification through logistic regression 
from sklearn.LinearModel import LogisticRegression 
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train , y_train)

#classification through kmeans algorithm 
from sklearn.neighbors import kNeighborsClassifier
classifier= kNeighborsClassifier( n_neighbors=5 , metric='minkowski' , p=2)
classifier.fit_transform(x_train , y_train)

#classification through SVM
from sklearn.svm import SVC
classifier=SVC(Kernel = 'linear', random_state=0)
classifier.fit_transform(x_train, y_train)

#classification through Navie Bayes
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred )

from matplotlib.colors import ListedColormap
x_set , y_set = x_test , y_test 
x1 , x2 = np.meshgrid(np.arrange(start = x_set[:,0].min() - 1 , stop= x_set[: , 0].max()+1 , step= 0.01), 
	                  np.arrange(start = x_set[:,1].min() - 1 , stop= x_set[: , 1].max()+1 , step= 0.01))
plt.contourf(x1, x2 , classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),alpha = 0.75 , cmap = ListedColormap(('red' ,'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate(np.unique(y_set)):
	plt.scatter(x_set[y_set == j,0] , x_set[y_set == j,1] , c = ListedColormap(('red','green'))(i), label=j)
plt.title('Logistic Regression:Test set ')
plt.xlabel('age')
plt.ylabel('est. salary')
plt.legend()
plt.show()

