import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
#define the parameters of the three blobs
n_samples = 300
centers = [[2,2],[-2,-2],[2,-2]]
cluster_std = [0.5,0.5,0.5]
#create the blobs
X,y = make_blobs(n_samples = n_samples,
				centers= centers,
				cluster_std=cluster_std,
				random_state= 42)
# Split data
X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.1,
	random_state=42
	)
# create the plot
plt.figure(figsize=(8,6))
#plot each blob with a different color
colors = ['#FF9999','#66B2FF','#99FF99']

for i in range(3):
	plt.scatter(
		X[y == i ,0],
		X[y == i,1],
		c=colors[i],
		label=f'blob{i+1}train',
		alpha = 0.6 )
# Draw TEST data
for i in range(3):
	plt.scatter(
		X_test[y_test == i, 0],
		X_test[y_test == i, 1],
		c=colors[i],
		marker='X',
		s=100,
		edgecolors='black',
		label=f'Blob {i+1} Test'
	)

# Customize plot
plt.title('Train and Test Gaussian Blobs')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# fit and perfrom k nearest neighbors model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
#Make perdictions
y_pred = knn.predict(X_test)
#calculate accuacy
accuracy = accuracy_score(y_test,y_pred)
print(f"Classification accuracy:{accuracy}")