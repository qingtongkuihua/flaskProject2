import importlib

import torch
from flask import Flask, request, jsonify
import os

from test_classification import parse_args, main, test

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # 获取输入数据

    # 检查请求是否包含文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    # 获取上传的文件
    file = request.files['file']

    # 检查文件类型是否为文本文件
    if file.content_type != 'text/plain':
        return jsonify({'error': 'Invalid file type'})

    # 保存文件到本地
    filename = file.filename
    print(filename)
    file.save(filename)

    # 读取文件内容
    # with open(filename, 'r') as f:
    #     content = f.read()

    result = []
    with open('data/dataSet/objectTypeName.txt', 'r') as f:
        for line in f:
            result.append(list(line.strip('\n').split(',')))
    print(len(result))

    # 执行模型推理
    num_class = 40
    print(num_class)
    model = importlib.import_module("classModel")

    classifier = model.get_model(num_class, normal_channel=True)

    print("========================")

    experiment_dir = 'log/classification/' + "model_log"
    device = torch.device('cpu')
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        pred = test(classifier.eval(), filename=filename, vote_num=1)

    print(pred.shape)

    # return "hellp"

    # 删除本地保存的文件
    os.remove(filename)
    # 返回输出数据
    return jsonify({'predict_result': result[pred.item()][0]})


if __name__ == '__main__':
    # 加载模型

    # 启动 Flask 应用
    app.run(host='0.0.0.0', port=5000)
