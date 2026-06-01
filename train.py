import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Đọc dữ liệu
df = pd.read_excel('Obesity_DataSet.xlsx')
print("Phân bố nhãn:\n", df['Class'].value_counts(normalize=True))
print("Phân bố Physical_Excercise:\n", df['Physical_Excercise'].value_counts(normalize=True))
print("Phân bố Type_of_Transportation_Used:\n", df['Type_of_Transportation_Used'].value_counts(normalize=True))

# Thêm đặc trưng tương tác
df['Activity_Level'] = df['Physical_Excercise'] + df['Type_of_Transportation_Used']

# Tách đặc trưng và nhãn, loại bỏ Age
X = df.drop(columns=['Class'])
y = df['Class']

# Cân bằng dữ liệu với SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

# Tạo và huấn luyện mô hình AdaBoost
base_estimator = DecisionTreeClassifier(max_depth=4)
model_adaboost = AdaBoostClassifier(estimator=base_estimator, n_estimators=200, learning_rate=1.0, random_state=42)
model_adaboost.fit(X_train_scaled, y_train)

# Đánh giá mô hình
y_pred = model_adaboost.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# In tầm quan trọng đặc trưng
features = X.columns
importances = model_adaboost.feature_importances_
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.4f}")

#Lưu cả mô hình và scaler vào 1 file
model_package = {
    'model': model_adaboost,
    'scaler': scaler
}
with open('obesity_adaboost_full.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("Đã lưu cả mô hình và scaler vào 1 file obesity_adaboost_full.pkl!")
