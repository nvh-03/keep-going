import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_1000_hoc_sinh.csv")

plt.hist(
    df["Điểm Trung Bình"],
    bins=20
)

plt.xlabel("Điểm Trung Bình")
plt.ylabel("Số học sinh")
plt.title("Phân bố điểm trung bình của 1.000 học sinh")

plt.grid()
plt.show()