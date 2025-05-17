from flask import Blueprint, request, render_template, current_app, redirect, url_for
from werkzeug.utils import secure_filename
from features_Extract import extract_pe_features
import os
import joblib
import time

choose_file_route = Blueprint('choose_file', __name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'exe'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to display the upload page and handle file uploads
@choose_file_route.route("/choose-file", methods=["GET", "POST"])
def choose_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            for attempt in range(3):
                try:
                    file.save(filepath)
                    break
                except OSError:
                    if attempt == 2:  # Last retry failed
                        return "File saving failed after multiple attempts", 500
                    time.sleep(1)  # Wait before retrying

            # Extract features
            features = extract_pe_features(filepath)
            if not features:
                return "Feature extraction failed", 400

            # Load the model and make prediction
            model = joblib.load('models/trained_model.pkl')
            prediction = model.predict([list(features.values())])[0]
            result = "Ransomware" if prediction == 1 else "Benign (safe exe)"

            # Render the template with prediction results
            return render_template('upload.html', result=result, features=features)

    # Render the file upload form
    return render_template('upload.html')
