from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)
model = load_model("Blood Cell.h5")
classes = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join('static', filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    result = classes[class_index]

    return render_template('result.html', prediction=result, image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
