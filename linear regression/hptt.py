import pandas as pd # doc faile csv
import numpy as np # vector and matric
import matplotlib.pyplot as plt  # ve bieu do phuong trinh duong thang
import seaborn as sns


#read in the dataset
df = pd.read_csv(".data.csv")
df.columns = ["index","height","weight"]
print(df.head())

# Create a scatterplot
sns.scatterplot(
	data=df,
	x =  "height",
	y = "weight")
plt.show()

# get X and Y
x = df["height"].values
y = df["weight"].values

#find m and b
N=x.shape[0]
m = (N*np.sum(x*y)-np.sum(x)*np.sum(y))/(N*np.sum(x**2)-(np.sum(x)**2))
b = (np.sum(y)-m*np.sum(x))/N
print("m:",m,"and b:",b)

#draw regression line
x_min = np.min(x)
y_min = m* x_min + b 
x_max = np.max(x)
y_max = m*x_max + b 

fig,ax = plt.subplots()
sns.scatterplot(
	data=df,
	x = "height",
	y = "weight",
	ax=ax,
	alpha = 0.4
)
sns.lineplot(
	x=[x_min,x_max],
	y=[y_min,y_max],
	linewidth=1.5,
	color='red'
)
plt.show()