import pandas as pd
import matplotlib.pyplot as plt

# Đọc file CSV
df = pd.read_csv("dataset_1000_hoc_sinh.csv")

# Các môn học
subjects = [
    "Điểm Toán",
    "Điểm Văn",
    "Điểm Tiếng Anh",
    "Điểm Lý",
    "Điểm Sinh",
    "Điểm Hóa"
]

# Tính điểm trung bình của từng môn
mean_scores = df[subjects].mean()

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))

plt.bar(
    mean_scores.index,
    mean_scores.values
)

# Tên trục
plt.xlabel("Môn học")
plt.ylabel("Điểm trung bình")

# Tiêu đề
plt.title("Điểm trung bình của từng môn")

# QUAN TRỌNG: đặt thang điểm từ 0 đến 10
plt.ylim(0, 10)

# Xoay tên môn
plt.xticks(rotation=45)

# Hiển thị grid
plt.grid(axis="y")

plt.tight_layout()
plt.show()