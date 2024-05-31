from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import io

app = Flask(__name__)

# 加载ONNX模型并创建推理会话
ort_session = ort.InferenceSession("output\\resnet18.onnx")

# 图像预处理函数
def preprocess(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    return input_tensor.unsqueeze(0).numpy()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image = Image.open(io.BytesIO(file.read()))
    input_data = preprocess(image)
    
    outputs = ort_session.run(None, {'input': input_data})
    
    # 这里假设模型的输出是一个分类概率
    # 如果是其他类型的输出，请根据需要调整处理
    predicted_class = np.argmax(outputs[0], axis=1)[0]
    
    return jsonify({"predicted_class": int(predicted_class)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
