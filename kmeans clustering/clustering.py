import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
##create a dataset
#create blobs
centers = np.array([
	[-5,-5],
	[-5, 5],
	[5,-5],
	[5, 5],
])
#generate 4 blobs based on the centers
blobs, groups = make_blobs(n_samples=200,n_features=2,centers=centers,cluster_std=1.5)
plt.scatter(blobs[:,0],blobs[:,1],c='gray')
plt.show()
#create random centroid
colors = ['r', 'b','g','y']
centroids = np.array([
	[-10,1],
	[-3,12],
	[10,10],
	[0,0],
	])
#plot the centroids
for i in range(4):
	plt.scatter(centroids[i:i+1,0],centroids[i:i+1,1],c=colors[i],s=250)
plt.scatter(blobs[:,0],blobs[:,1],c='gray')
plt.show()
for i in range(200):
	#step1: allocate the points based on the centroid
	allocations=[]
	for j in range (blobs.shape[0]):
		distances = np.sum((blobs[j] - centroids) ** 2, axis=1)
		argmin = np.argmin(distances)
		allocations.append(argmin)
	#convert to numpy
	allocations = np.array(allocations)
	#calculate the centroid
	for j in range (centroids.shape[0]):
		centroids[j]=blobs[allocations==j,:].mean(axis=0)
#create random centroid
colors = ['r','b','g','y']
#plot the centroids
plt.scatter(blobs[:,0],blobs[:,1],c='gray')
for i in range(4):
	plt.scatter(centroids[i:i+1,0],centroids[i:i+1,1],c=colors[i],s=250)
plt.show()


