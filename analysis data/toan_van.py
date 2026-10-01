import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_1000_hoc_sinh.csv")

plt.scatter(
    df["Điểm Toán"],
    df["Điểm Văn"]
)

plt.xlabel("Điểm Toán")
plt.ylabel("Điểm Văn")
plt.title("Mối quan hệ giữa điểm Toán và Văn")

plt.grid()
plt.show()