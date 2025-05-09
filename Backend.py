import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
from flask_cors import CORS
from flask_mail import Mail, Message
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import smtplib
import random
import matplotlib.pyplot as plt


app = Flask(__name__)
CORS(app)


app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB


app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URI", "mysql+pymysql://root:@localhost/eyecardia")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your_secret_key")

print(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")


db = SQLAlchemy(app)


app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'eyecardia@gmail.com'
app.config['MAIL_PASSWORD'] = 'hzls oozk zbzz qvys'


mail = Mail(app)


login_manager = LoginManager()
login_manager.init_app(app)



class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(512), nullable=False)
    otp = db.Column(db.String(6), nullable=True)
    category = db.Column(db.String(50))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



classification_model = tf.keras.models.load_model(r"C:\Users\Yuktha Varshika\Music\Models\Retinal_Disease_Classification.h5")
severity_model = tf.keras.models.load_model(r"C:\Users\Yuktha Varshika\Music\Models\Retinal_Severity_Classification.h5")


class_labels = ["Age-Related Macular Degeneration", "Diabetic Retinopathy", "Hypertensive Retinopathy", "Retinal Artery Occlusion", "Retinal Vein Occlusion"]
severity_labels = ["Mild", "Moderate", "Severe"]

disease_info = {
    "Age-Related Macular Degeneration": {"description": "Affects central vision due to macular deterioration.", "diagnosis": "Consult an ophthalmologist for treatments like anti-VEGF injections or laser therapy."},
    "Diabetic Retinopathy": {"description": "Complication of diabetes causing damage to retinal blood vessels.", "diagnosis": "Control blood sugar, undergo regular eye check-ups, and consider laser therapy."},
    "Hypertensive Retinopathy": {"description": "Results from prolonged high blood pressure affecting the retina.", "diagnosis": "Monitor blood pressure and visit an eye specialist."},
    "Retinal Artery Occlusion": {"description": "Blocked arteries leading to sudden vision loss.", "diagnosis": "Immediate medical attention is required, with treatments like blood thinners."},
    "Retinal Vein Occlusion": {"description": "Blocked veins leading to retinal swelling and vision loss.", "diagnosis": "Regular eye check-ups, anti-VEGF therapy, or laser treatment may be needed."}
}


def classify_severity(confidence):
    if confidence > 0.8:
        return "Mild-60%", "Regular check-ups recommended."
    elif confidence > 0.5:
        return "Moderate-75%", "Consult an ophthalmologist for further evaluation."
    else:
        return "Severe-85%", "Immediate medical attention required."


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=9.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)


def gamma_correction(image, gamma=1.5):
    image = image.astype(np.float32) / 255.0
    corrected_image = np.power(image, gamma)
    return np.clip(corrected_image * 255, 0, 255).astype(np.uint8)


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_clahe = apply_clahe(image)
    image_corrected = gamma_correction(image_clahe, gamma=1.5)
    
    final_image = cv2.resize(image_corrected, (224, 224)) / 255.0
    return np.expand_dims(final_image, axis=0)


def classify_retinal_disease(image_path):
    image = preprocess_image(image_path)
    
    prediction = classification_model.predict(image)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    confidence = prediction[0][predicted_class]
    
    if predicted_label == "Diabetic Retinopathy":
        severity_prediction = severity_model.predict(image)
        severity_class = np.argmax(severity_prediction)
        severity_label = severity_labels[severity_class]
    else:
        severity_label = "Mild" if confidence > 0.8 else "Moderate" if confidence > 0.5 else "Severe"
    
    return {
        "disease": predicted_label,
        "confidence": float(confidence),
        "severity": severity_label,
        "description": disease_info[predicted_label]["description"],
        "diagnosis": disease_info[predicted_label]["diagnosis"]
    }


@app.route("/")
def home():
    return "Welcome to the EyeCardia Prediction App!"


@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    print("Received data:", data)  # Debugging line

    email = data.get("email")
    category = data.get("category")  # Check what value is received

    if not email or not category:
        return jsonify({"error": "Missing email or category"}), 400

   
    print(f"Email: {email}, Category: {category}")

    
    valid_categories = {
        "Cardiologists & Stroke Specialists": "Cardiologists & Stroke Specialists",
        "Diabetic & Hypertensive Patients": "Diabetic & Hypertensive Patients",
        "Optometrists & Eye Clinics": "Optometrists & Eye Clinics"
    }
    
   
    if category in valid_categories:
        category = valid_categories[category]  

    
    if category not in valid_categories.values():
        return jsonify({"error": "Invalid category"}), 400

    
    user = User.query.filter_by(email=email).first()
    if user:
        return jsonify({"error": "User already exists"}), 409

    
    otp = str(random.randint(100000, 999999))

    
    new_user = User(email=email, otp=otp, category=category)
    db.session.add(new_user)
    db.session.commit()

    
    try:
        msg = Message("Your OTP Code", sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.body = f"Your OTP code is: {otp}"
        mail.send(msg)
    except Exception as e:
        print(f"Email sending failed: {e}")
        return jsonify({"error": "Failed to send OTP"}), 500

    return jsonify({"message": "OTP sent successfully!"}), 201



from werkzeug.security import generate_password_hash, check_password_hash

@app.route("/validate_otp", methods=["POST"])
def validate_otp():
    try:
        data = request.get_json()
        email = data.get("email", "").strip()
        otp = data.get("otp", "").strip()
        password = data.get("password", "").strip()
        confirm_password = data.get("confirm_password", "").strip()

        if not email or not otp or not password or not confirm_password:
            return jsonify({"error": "All fields are required"}), 400

        if password != confirm_password:
            return jsonify({"error": "Passwords don't match"}), 400

        
        user = User.query.filter_by(email=email, otp=otp).first()
        if not user:
            return jsonify({"error": "Invalid OTP/Email"}), 400

       
        hashed_password = generate_password_hash(password)  
        print(f"🔑 Hashed Password Before Storing in DB: {hashed_password}")

        user.password = hashed_password
        user.otp = None  
        db.session.commit()

        print("✅ Registration successful! Hashed password stored.")
        return jsonify({"message": "Registration successful"}), 200

    except Exception as e:
        print(f"⚠ Registration error: {e}")
        return jsonify({"error": "Registration failed"}), 500


@app.route("/request_password_reset", methods=["POST"])
def request_password_reset():
    data = request.get_json()
    email = data.get("email", "").strip()

    if not email:
        return jsonify({"success": False, "error": "Email is required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404

    otp = str(random.randint(100000, 999999))
    user.otp = otp
    db.session.commit()

    try:
        msg = Message("Password Reset OTP", sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.body = f"Use this OTP to reset your password: {otp}"
        mail.send(msg)
    except Exception as e:
        print(f"Error sending OTP: {e}")
        return jsonify({"success": False, "error": "Failed to send OTP"}), 500

    return jsonify({"success": True, "message": "OTP sent to your email"}), 200





@app.route("/reset_password", methods=["POST"])
def reset_password():
    data = request.get_json()
    otp = data.get("otp", "").strip()
    new_password = data.get("new_password", "").strip()
    confirm_password = data.get("confirm_password", "").strip()

    if not all([otp, new_password, confirm_password]):
        return jsonify({"success": False, "error": "All fields are required"}), 400

    if new_password != confirm_password:
        return jsonify({"success": False, "error": "Passwords do not match"}), 400

    user = User.query.filter_by(otp=otp).first()
    if not user:
        return jsonify({"success": False, "error": "Invalid OTP"}), 400

    hashed_password = generate_password_hash(new_password)
    user.password = hashed_password
    user.otp = None
    db.session.commit()

    return jsonify({"success": True, "message": "Password has been reset successfully"}), 200


@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400  

        email = data.get("email", "").strip()
        password = data.get("password", "").strip()

        if not email or not password:
            return jsonify({"error": "Email and Password are required"}), 400  

        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401  

        if check_password_hash(user.password, password):
            return jsonify({"success": True, "message": "Login successful"}), 200 
        else:
            return jsonify({"error": "Invalid credentials"}), 401  

    except Exception as e:
        print(f"⚠ Login error: {e}")
        return jsonify({"error": "Server error. Please try again later."}), 500  


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:

        file.seek(0, 2)  
        file_size = file.tell()  
        file.seek(0)  
        print(f"Uploaded file size: {file_size} bytes")

        if file_size > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({"error": f"File size exceeds the maximum allowed limit ({app.config['MAX_CONTENT_LENGTH']} bytes)"}), 413

 
        image_path = os.path.join("uploads", file.filename)
        file.save(image_path)

        model_input = preprocess_image(image_path)


        print("Input to model shape:", model_input.shape)


        predictions = classification_model.predict(model_input)
        predicted_label = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions)

    
        if predicted_label not in class_labels:
            return jsonify({
                "Predicted Disease": "Normal",
                "Description": "No retinal disease detected.",
                "Severity Level": "None",
                "Medical Suggestions": "Routine follow-up recommended."
            })
        
        severity, recommendation = classify_severity(confidence)
        predicted_description = disease_info[predicted_label]["description"]

        return jsonify({
            "Predicted Disease": predicted_label,
            "Description": predicted_description,
            "Severity Level": severity,
            "Medical Suggestions": recommendation,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File size exceeds the maximum allowed limit (200 MB)"}), 413


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
