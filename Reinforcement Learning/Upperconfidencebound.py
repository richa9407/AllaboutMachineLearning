import numpy as numpy
import matplotlib.pyplot as plt
import pandas as pd

#ads_CTR_optimization dataset 
dataset = pd.read_csv('/Users/savita/desktop/AllaboutMachineLearning/Reinforcement Learning/Ads_CTR_Optimisation.csv')

import math 
N= 10000
d= 10 
ads_selected = []
numbers_of_selections = [0]*d
sum_of_rewards = [0]*d
total_rewards = 0
for n in range(0,N):
	ad= 0 
	max_upper_bound = 0
	for i in range(0,d):
		if (numbers_of_selections[i]>0 ):
			average_reward = sum_of_rewards[i] / numbers_of_selections[i]
			delta_i = math.sqrt(3/2 *math.log(n+1)/numbers_of_selections[i])
			upper_bound = average_reward + delta_i
		else:
			upper_bound= 1e400
		if upper_bound> max_upper_bound:
			max_upper_bound= upper_bound
			ad = i
	ads_selected.append(ad)
	numbers_of_selections[ad] = numbers_of_selections[ad] +1
	reward = dataset.values[n,ad]
	sum_of_rewards[ad] =sum_of_rewards[ad] +reward
	total_rewards= total_rewards+ reward
			

plt.hist(ads_selected)
plt.title('Histogram of ads selection ')
plt.xlabel('ads')
plt.ylabel('number of items each ad was selected')
plt.show()
