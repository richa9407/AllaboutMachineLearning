import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

dataset = pd.read_csv('/Users/savita/desktop/AllaboutMachineLearning/clustering /Mall_Customers.csv')
x= dataset.iloc[: , [3,4]].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('Euclidian distances')
plt.show()

#fitting heirarchial clustering to the dataset 
from sklearn.cluster import AgglomerativeClustering 
clustering= AgglomerativeClustering(n_clusters=5 , affinity= 'euclidean', linkage='ward')
y_hc= clustering.fit_predict(x)

#plot the sho
plt.scatter(x[y_hc==0,0],x[y_hc==0,1], s=100 , c='red', label = 'Cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1], s=100 , c='magenta', label = 'Cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1], s=100 , c='green', label = 'Cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1], s=100 , c='blue', label = 'Cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1], s=100 , c='cyan', label = 'Cluster 5')
plt.title('Clusters of classifiers')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()


