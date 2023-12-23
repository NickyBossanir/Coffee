from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
import numpy as np
import joblib

app = Flask(__name__)

# 加载模型

model1 = joblib.load('model1.pkl')
model2 = joblib.load('model2.pkl')
model3 = joblib.load('model3.pkl')
model4 = joblib.load('model4.pkl')

@app.route('/')
def index():
    return render_template('Input.html')

@app.route('/Input', methods=['POST'])
def Input():
    # 从HTML表单中获取数据
    input_data = {
        '豆の量': float(request.form['豆の量']),
        'お湯の温度': float(request.form['お湯の温度']),
        '抽出量': float(request.form['抽出量']),
        'ミルのクリック数': float(request.form['ミルのクリック数']),
        '蒸らし時間': float(request.form['蒸らし時間']),
        # 添加其他特征...
    }

    # 将输入数据转换为DataFrame
    input_df = pd.DataFrame([input_data])

    # 使用模型进行预测
    prediction1 = model1.predict(np.array([[input_data['豆の量'], input_data['お湯の温度'], input_data['抽出量'], input_data['ミルのクリック数'], input_data['蒸らし時間']]]))
    prediction2 = model2.predict(input_df.values)
    prediction3 = model3.predict(input_df.values)
    prediction4 = model4.predict(input_df.values)

    # 将预测结果传递回HTML页面
    return render_template('Output.html', prediction1=prediction1, prediction2=prediction2, prediction3=prediction3, prediction4=prediction4)

if __name__ == '__main__':
    app.run(debug=True)