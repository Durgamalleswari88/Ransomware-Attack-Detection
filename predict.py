import os
import joblib
from werkzeug.utils import secure_filename
from flask import Blueprint, request, render_template, current_app
from features_Extract import extract_pe_features

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'exe'}

# Blueprint for predict routes
predict_bp = Blueprint('predict_bp', __name__)


# Function to check if the file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to handle file upload and prediction
@predict_bp.route('/predict', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists
            file.save(filepath)

            # Extract features from the uploaded file
            features = extract_pe_features(filepath)
            if not features:
                return "Feature extraction failed", 400

            # Load the trained model
            model = joblib.load('models/trained_model.pkl')

            # Make prediction
            prediction = model.predict([list(features.values())])[0]
            result = "Ransomware" if prediction == 1 else "Benign (safe exe)"

            # Render the result page with the prediction
            return render_template('prediction.html', result=result, features=features)

    return render_template('predict.html')  # Render the file upload form
