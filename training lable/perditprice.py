import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# =========================
# 1. Đọc dữ liệu
# =========================
house = pd.read_csv("house.csv")


# =========================
# 2. Chọn features và label
# =========================
def select_features(df, feature_names):
    return df[feature_names].values


def select_label(df, label_name):
    return df[label_name].values


features = select_features(
    house,
    ["area", "bedrooms", "bathrooms"]
)

label = select_label(
    house,
    "price"
)


# =========================
# 3. Chia dữ liệu
# =========================
features_train_val, features_test, label_train_val, label_test = train_test_split(
    features,
    label,
    test_size=0.1,
    random_state=42
)

features_train, features_val, label_train, label_val = train_test_split(
    features_train_val,
    label_train_val,
    test_size=1/9,
    random_state=42
)


# =========================
# 4. Train model
# =========================
model = LinearRegression()

model.fit(
    features_train,
    label_train
)


# =========================
# 5. Đánh giá model
# =========================
train_r2_score = model.score(
    features_train,
    label_train
)

val_r2_score = model.score(
    features_val,
    label_val
)

print("R2 train:", train_r2_score)
print("R2 validation:", val_r2_score)


# =========================
# 6. Xem công thức model
# =========================
#print("Area:", model.coef_[0])
#print("Bedrooms:", model.coef_[1])
#print("Bathrooms:", model.coef_[2])

#print("Intercept:", model.intercept_)


# =========================
# 7. Dự đoán nhà mới
# =========================

area = float(input("\nNhap dien tich nha (area): "))
bedrooms = float(input("Nhap so phong ngu: "))
bathrooms = float(input("Nhap so phong tam: "))


new_house = [[
    area,
    bedrooms,
    bathrooms
]]


predicted_price = model.predict(new_house)


print("\n========================")
print("Dien tich:", area)
print("Phong ngu:", bedrooms)
print("Phong tam:", bathrooms)
print("Gia nha du doan:", predicted_price[0])
print("========================")