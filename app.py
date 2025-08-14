import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = (
    "0"  # Nonaktifkan optimasi oneDNN untuk menghindari masalah kompatibilitas
)
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, redirect

# Local Libraries
from src.predict import predict_image
from src.constants import UPLOAD_FOLDER, ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    """Fungsi untuk memeriksa ekstensi file yang diizinkan."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            prediction_class, probabilities = predict_image(file_path)
            
            print(f"prediction_class: {prediction_class}")
            print(probabilities)
            for item in probabilities:
                print(item["label"])
                print(f"Score: {item['score']}")
                print(f"Kalian: {'{:.17f}'.format(item['score'])}")
            return render_template(
                "index.html",
                filename=filename,
                prediction=prediction_class,
                probabilities=probabilities,
            )
    return render_template("index.html")


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
