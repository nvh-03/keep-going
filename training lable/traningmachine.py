import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn

from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression

house = pd.read_csv("house.csv")#doc file
#print(house.head())
#select features and lables
def select_features (df,feature_names):
	return df[feature_names].values

def select_label(df,lable_name):
	return df[lable_name].values
features = select_features(house,["area","bedrooms","bathrooms","stories"])
label = select_label(house,"price")
#print(feature)
# 90% train + validation
# 10% test
features_train_val, features_test, label_train_val, label_test = train_test_split(
	features,
	label,
	test_size=0.1,
	random_state=42
)
# Từ 90% còn lại:
# 80% train
# 10% validation
features_train, features_val, label_train, label_val = train_test_split(
	features_train_val,
	label_train_val,
	test_size=1/9,
	random_state=42
)
model = LinearRegression()

model.fit(features_train,label_train)
train_r2_score = model.score(features_train,label_train)
val_r2_score = model.score(features_val,label_val)
print(f"R2 score of train set :{train_r2_score}")
print(f"R2 score of val set :{val_r2_score}")	