from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd

app = Flask(__name__)

# 加载模型
model = xgb.XGBClassifier()
model.load_model("checkpoint\\xgboost_iris_model.json")

@app.route('/predict', methods=['POST'])
def predict():
    # 从请求中获取数据
    data = request.get_json(force=True)
    df = pd.DataFrame(data)

    # 进行预测
    predictions = model.predict(df)
    prediction_labels = predictions.tolist()

    # 返回结果
    return jsonify(prediction=prediction_labels)

if __name__ == '__main__':
    app.run(debug=True)
