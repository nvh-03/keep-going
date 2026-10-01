import pandas as pd
import matplotlib.pyplot as plt

# Đọc file CSV
df = pd.read_csv("dataset_1000_hoc_sinh.csv")

# Lấy 10 học sinh đầu tiên
data = df.head(10)

# Vẽ biểu đồ
plt.plot(data["Tên"], data["Điểm Toán"], marker="o")

plt.xlabel("Học sinh")
plt.ylabel("Điểm Toán")
plt.title("Điểm Toán của 10 học sinh đầu tiên")

plt.xticks(rotation=20)
plt.ylim(0, 10)
plt.tight_layout()
plt.show()