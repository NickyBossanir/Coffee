import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from flask import Flask, render_template, request
import numpy as np

# 读取Excel文件
file_path = 'Data/Data.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(file_path)

# 分离自变量和因变量
X = df.iloc[:, :-4]  # 选择所有行和除最后一列外的所有列作为自变量
y1 = df.iloc[:,-4]   # 选择所有行和最后一列作为因变量
y2 = df.iloc[:,-3]
y3 = df.iloc[:,-2]
y4 = df.iloc[:,-1]
# 将数据分为训练集和测试集
X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test = train_test_split(X, y1,y2,y3,y4, test_size=0.2, random_state=42)

# 创建线性回归模型
model1 = LinearRegression()
model2 = LinearRegression()
model3 = LinearRegression()
model4 = LinearRegression()
# 将模型拟合到训练数据上
model1.fit(X_train, y1_train)
model2.fit(X_train, y2_train)
model3.fit(X_train, y3_train)
model4.fit(X_train, y4_train)

# 在测试集上进行预测
y1_pred = model1.predict(X_test)
y2_pred = model2.predict(X_test)
y3_pred = model3.predict(X_test)
y4_pred = model4.predict(X_test)

# 计算模型的均方误差（MSE）
mse1 = mean_squared_error(y1_test, y1_pred)
print(f'Mean Squared Error: {mse1}')
mse2 = mean_squared_error(y2_test, y2_pred)
print(f'Mean Squared Error: {mse2}')
mse3 = mean_squared_error(y3_test, y3_pred)
print(f'Mean Squared Error: {mse3}')
mse4 = mean_squared_error(y4_test, y4_pred)
print(f'Mean Squared Error: {mse4}')

# 输出模型的系数和截距
print('Coefficients:', model1.coef_)
print('Intercept:', model1.intercept_)
print('Coefficients:', model2.coef_)
print('Intercept:', model2.intercept_)
print('Coefficients:', model3.coef_)
print('Intercept:', model3.intercept_)
print('Coefficients:', model4.coef_)
print('Intercept:', model4.intercept_)


# 保存模型为.pkl文件
joblib.dump(model1, 'model1.pkl')
joblib.dump(model2, 'model2.pkl')
joblib.dump(model3, 'model3.pkl')
joblib.dump(model4, 'model4.pkl')