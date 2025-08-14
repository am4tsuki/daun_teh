import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Local constant
CLASS_LABELS = ["Matang", "Muda", "Tua"]

models_file = "models/cnn_mobilenetv2_model_lts.h5"
try:
    model = load_model(models_file, compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

print(f"Urutan kelas yang digunakan: {CLASS_LABELS}")


def predict_image(file_path):
    """Fungsi untuk memproses gambar dan melakukan prediksi."""
    img = image.load_img(file_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0]

    print(f"Prediction: {prediction}")

    class_index = np.argmax(prediction)
    class_name = CLASS_LABELS[class_index]
    probabilities = []
    for i, label in enumerate(CLASS_LABELS):
        prob_score = prediction[i]
        probabilities.append(
            {
                "label": label,
                "score": float(prob_score),
            }
        )

    probabilities.sort(key=lambda x: x["score"], reverse=True)

    return class_name, probabilities
