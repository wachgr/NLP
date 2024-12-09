import numpy as np
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
import pandas as pd
import random
# n=1
# x=np.random.uniform(1,2,1000)
# y = x.copy()
# x = x - np.mean(x)
# y = y - np.mean(y)
# data = pd.DataFrame({'x':x,'y':y})
# plt.scatter(data.x,data.y)
# pca = PCA(n_components=2)
# pcaTr = pca.fit(data)
# print(pcaTr)
# rotatedData = pcaTr.transform(data)
# dataPCA = pd.DataFrame(data = rotatedData, columns = ['PC1', 'PC2'])
random.seed(100)

std1 = 1     # The desired standard deviation of our first random variable
std2 = 0.333 # The desired standard deviation of our second random variable

x = np.random.normal(0, std1, 1000) # Get 1000 samples from x ~ N(0, std1)
y = np.random.normal(0, std2, 1000)  # Get 1000 samples from y ~ N(0, std2)

x = x - np.mean(x) # Center x
y = y - np.mean(y) # Center y


n = 1
angle = np.arctan(1 / n)
print('angle: ',  angle )
rotationMatrix = np.array([[np.cos(angle), np.sin(angle)],
                 [-np.sin(angle), np.cos(angle)]])
print('rotationMatrix')
print(rotationMatrix)

xy = np.concatenate(([x] , [y]), axis=0).T
data = np.dot(xy, rotationMatrix)
plt.scatter(data[:,0], data[:,1])
plt.show()

