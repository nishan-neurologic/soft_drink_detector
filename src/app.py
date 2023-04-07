# from flask import Flask, request, send_file, jsonify, make_response, render_template
# from werkzeug.utils import secure_filename
# import os
# import cv2
# import tempfile

# from predict import predict_image_flask, configure_predictor  # Assuming your inference code is in 'predict.py'

# app = Flask(__name__)

# from flask import Flask, request, send_file, jsonify, make_response
# from werkzeug.utils import secure_filename
# import os
# import cv2
# import tempfile
# import base64
# import json
# from predict import predict_image, configure_predictor
# from detectron2.data import MetadataCatalog, DatasetCatalog

# # Load the metadata for your dataset
# metadata = MetadataCatalog.get("my_dataset_train")

# from detectron2.data import DatasetCatalog, MetadataCatalog

# def your_dataset_function():
#     dataset_dicts = json.loads('/Users/nishanali/WorkSpace/rani-peach/data/test/_annotations.coco.json')
#     return dataset_dicts
# # Add metadata for your dataset
# MetadataCatalog.get("my_dataset_train").set(thing_classes=["Rani-Peach", "RaniPeachrotation"])
# MetadataCatalog.get("my_dataset_val").set(thing_classes=["Rani-Peach", "RaniPeachrotation"])
# MetadataCatalog.get("my_dataset_test").set(thing_classes=["Rani-Peach", "RaniPeachrotation"])

# # Register your dataset in the DatasetCatalog
# DatasetCatalog.register("your_dataset_name", your_dataset_function)

# # Add metadata for your dataset
# MetadataCatalog.get("your_dataset_name").set(thing_classes=["class0", "class1"])

# # Register your dataset in the DatasetCatalog
# DatasetCatalog.register("my_dataset_train", lambda: your_dataset_function())

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     if 'file' not in request.files:
# #         return jsonify({'error': 'No file part in the request'}), 400

# #     file = request.files['file']
# #     if file.filename == '':
# #         return jsonify({'error': 'No file selected for uploading'}), 400

# #     filename = secure_filename(file.filename)
# #     temp_dir = tempfile.mkdtemp()
# #     file_path = os.path.join(temp_dir, filename)
# #     file.save(file_path)

# #     cfg = configure_predictor()
# #     output_image_path = predict_image_flask(cfg, file_path)
# #     os.unlink(file_path)  # Clean up the input file

# #     # Read the output image and encode it as base64
# #     with open(output_image_path, "rb") as image_file:
# #         encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# #     response = make_response(jsonify({'image': encoded_image}))
# #     response.headers.set('Content-Type', 'application/json')

# #     return response

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part in the request'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected for uploading'}), 400

#     filename = secure_filename(file.filename)
#     temp_dir = tempfile.mkdtemp()
#     file_path = os.path.join(temp_dir, filename)
#     file.save(file_path)

#     cfg = configure_predictor()
#     output_image_path, label_info = predict_image_flask(cfg, file_path)
#     os.unlink(file_path)  # Clean up the input file

#     # Read the output image and encode it as base64
#     with open(output_image_path, "rb") as image_file:
#         encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

#     response_data = {
#         'image': encoded_image,
#         'label_info': label_info
#     }
#     response = make_response(jsonify(response_data))
#     response.headers.set('Content-Type', 'application/json')

#     return response

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify, make_response, render_template
from werkzeug.utils import secure_filename
import os
import tempfile
import base64
import json
from predict import predict_image, configure_predictor, predict_image_flask
from detectron2.data import MetadataCatalog, DatasetCatalog

app = Flask(__name__)

def your_dataset_function():
    dataset_dicts = json.loads('../data/train/_annotations.coco.json')
    return dataset_dicts

# Register your dataset in the DatasetCatalog
DatasetCatalog.register("my_dataset_train", your_dataset_function)

# Add metadata for your dataset
MetadataCatalog.get("my_dataset_train").set(thing_classes=["Rani-Peach", "RaniPeachrotation"])
MetadataCatalog.get("my_dataset_val").set(thing_classes=["Rani-Peach", "RaniPeachrotation"])
MetadataCatalog.get("my_dataset_test").set(thing_classes=["Rani-Peach", "RaniPeachrotation"])



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    filename = secure_filename(file.filename)
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)
    file.save(file_path)

    cfg = configure_predictor()
    output_image_path, label_info = predict_image_flask(cfg, file_path)
    os.unlink(file_path)  # Clean up the input file

    # Read the output image and encode it as base64
    with open(output_image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    response_data = {
        'image': encoded_image,
        'label_info': label_info
    }
    response = make_response(jsonify(response_data))
    response.headers.set('Content-Type', 'application/json')

    return response

if __name__ == '__main__':
    app.run()
