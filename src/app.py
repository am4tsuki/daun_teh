import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Nonaktifkan optimasi oneDNN untuk menghindari masalah kompatibilitas
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

models_file = 'models/cnn_mobilenetv2_model.h5'

try:
    model = load_model(models_file, compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

CLASS_LABELS = ['Matang', 'Muda', 'Tua']
# CLASS_LABELS.sort()
print(f"Urutan kelas yang digunakan: {CLASS_LABELS}")


def allowed_file(filename):
    """Fungsi untuk memeriksa ekstensi file yang diizinkan."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(file_path):
    """Fungsi untuk memproses gambar dan melakukan prediksi."""
    img = image.load_img(file_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0]
    
    class_index = np.argmax(prediction)
    class_name = CLASS_LABELS[class_index]
    probabilities = []
    for i, label in enumerate(CLASS_LABELS):
        prob_score = prediction[i]
        probabilities.append({
            'label': label,
            'score': float(prob_score),
        })
        
    probabilities.sort(key=lambda x: x['score'], reverse=True)
    
    return class_name, probabilities

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            if model:
                prediction_class, probabilities = predict_image(file_path)
                print(f"prediction_class: {prediction_class}")
                print(probabilities)
                for item in probabilities:
                    print(item["label"])
                    print(f"Score: {item['score']}")
                    print(f"Kalian: {'{:.17f}'.format(item['score'])}")
                return render_template(
                    'index.html', 
                    filename=filename, 
                    prediction=prediction_class,
                    probabilities=probabilities # Kirim data probabilitas yang sudah diurutkan
                )
            else:
                return render_template('index.html', error="Model tidak dapat dimuat.")
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)