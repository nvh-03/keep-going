import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_1000_hoc_sinh.csv")

# Tìm những người có tên chứa "Long"
result = df[df["Tên"].str.contains("Long", case=False, na=False)]

print(result)
print("Có", len(result), "người tên Long")

plt.show()