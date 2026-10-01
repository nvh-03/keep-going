import numpy as np
import matplotlib.pyplot as plt 
dataset = np.array([
	[1,1,0],[2,1,0],[3,1,0],[4,1,0],
	[1,2,0],[2,2,0],[3,2,0],[4,2,0],
	[1,3,1],[2,3,1],[3,3,0],[4,3,0],
	[1,4,1],[2,4,1],[3,4,0],[4,4,0],
	[1,5,0],[2,5,0],[3,5,0],[4,5,0],
	[1,6,0],[2,6,0],[3,6,0],[4,6,0],
	])
#split the dataset into two groups
group_a = dataset[dataset[:,2]==0]
group_b = dataset[dataset[:,2]==1]
#draw group a with color blue
#plt.scatter(group_a[:,0],group_a[:,1],color='blue',s=100,label ='Group A')
#plt.scatter(group_b[:,0],group_b[:,1],color='red',s=100,label ='Group B')

from sklearn.tree import DecisionTreeClassifier,plot_tree
#Create inputs and labels
X = dataset[:, :2]
Y = dataset[:, 2]
#Create a decision tree object
decision_tree = DecisionTreeClassifier(criterion="gini")
decision_tree.fit(X,Y)
# Plot the tree
plt.figure(figsize=(12,6))
plt.subplot(1,2,2)
plot_tree(decision_tree,fontsize=10)
# Draw group a with color blue
plt.subplot(1,2,1)
plt.scatter(group_a[:,0],group_a[:,1],color='blue',s=100,label='Group A')
# Draw group a with color red
plt.scatter(group_b[:,0],group_b[:,1],color='red',s=100,label='Group B')
#show the plot
plt.title("Mouse dataset")
plt.xlabel("front legs'length")
plt.ylabel("back legs'length")
plt.legend()
plt.show()
