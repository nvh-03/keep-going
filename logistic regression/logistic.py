import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('ggplot') # set up styling for matplotlib

#increate a dataset
dataset = np.array ([
	[-10, 0],
	[-5, 0],
	[7, 1],
	[0, 0],
	[-2, 0],
	[2, 1],
	[5, 1],
	[-7, 0],
	[10, 1],
	[15, 1],
	[11, 1],
	[13, 1],
])
#negative_class = dataset[:5] #[:5] nghĩa là lấy 5 dòng đầu tiên:
#positive_class = dataset[5:]# [5:] 5 to end
# Tách class dựa vào cột thứ 2
negative_class = dataset[dataset[:, 1] == 0]
positive_class = dataset[dataset[:, 1] == 1]
# draw the negative first and positive
plt.scatter(negative_class[:,0],negative_class[:,1],c='y')
plt.scatter(positive_class[:,0],positive_class[:,1],c='g')
# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# Cost function
def get_cost(y, y_hat):
    epsilon = 1e-10

    cost = -np.mean(
        y * np.log(y_hat + epsilon)
        + (1 - y) * np.log(1 - y_hat + epsilon)
    )

    return cost

# Accuracy
def get_accuracy(y, y_hat):
    prediction = (y_hat >= 0.5).astype(int)

    accuracy = np.mean(prediction == y)

    return accuracy
# Gradient
def get_gradient(m, b, x, y, y_hat):

    dm = np.mean((y_hat - y) * x)

    db = np.mean(y_hat - y)

    return dm, db



# create the gradient descent flow
m = 1.0
b = 10.0
iterations = 150
lr = 0.03
x = dataset[:,0]
y = dataset[:,1]
costs = []
for it in range(iterations):
	 # 1. Tính z = mx + b
	z = m * x + b

    # 2. Tính prediction bằng sigmoid
	y_hat = sigmoid(z)

    # 3. Tính cost
	cost = get_cost(y, y_hat)

    # 4. Tính accuracy
	accuracy = get_accuracy(y, y_hat)

    # 5. Tính gradient
	dm, db = get_gradient(m, b, x, y, y_hat)

    # 6. Update m và b
	m -= lr * dm
	b -= lr * db

    # 7. Lưu cost
	costs.append(cost)

	print(
		f"iteration {it} - cost: {cost:.4f}, "
 		f"accuracy: {accuracy:.4f}"
	)


# =========================
# Vẽ đường phân loại
# =========================

x_line = np.linspace(-12, 17, 100)

z_line = m * x_line + b
y_line = sigmoid(z_line)

plt.plot(x_line, y_line)

plt.xlabel("x")
plt.ylabel("Probability")
plt.title("Logistic Regression using Gradient Descent")

plt.show()


# =========================
# Vẽ Cost
# =========================

plt.plot(costs)

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost during Gradient Descent")

plt.show()

print("Final m =", m)
print("Final b =", b)