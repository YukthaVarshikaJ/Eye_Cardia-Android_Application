import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

class_labels = ["Age-Related Macular Degeneration", "Diabetic Retinopathy", "Hypertensive Retinopathy", "Retinal Artery Occlusion", "Retinal Vein Occlusion"]
severity_labels = ["Mild", "Moderate", "Severe"]

disease_info = {
    "Age-Related Macular Degeneration": {
        "description": "Age-related macular degeneration (ARMD) affects central vision due to macular deterioration.",
        "diagnosis": "Consult an ophthalmologist. Treatments include anti-VEGF injections or laser therapy."
    },
    "Diabetic Retinopathy": {
        "description": "Diabetic Retinopathy (DR) is a complication of diabetes causing damage to the retinal blood vessels.",
        "diagnosis": "Control blood sugar, undergo regular eye check-ups, and consider laser therapy if necessary."
    },
    "Hypertensive Retinopathy": {
        "description": "Hypertensive Retinopathy (HR) results from prolonged high blood pressure affecting the retina.",
        "diagnosis": "Monitor blood pressure and visit an eye specialist for further evaluation."
    },
    "Retinal Artery Occlusion": {
        "description": "Retinal Artery Occlusion (RAO) occurs due to blocked arteries, leading to sudden vision loss.",
        "diagnosis": "Immediate medical attention is required. Treatment may involve blood thinners or clot-dissolving agents."
    },
    "Retinal Vein Occlusion": {
        "description": "Retinal Vein Occlusion (RVO) is caused by blocked veins, leading to retinal swelling and vision loss.",
        "diagnosis": "Regular eye check-ups, anti-VEGF therapy, or laser treatment may be needed."
    }
}

def load_models(classification_model_path, severity_model_path):
    classification_model = tf.keras.models.load_model(classification_model_path)
    severity_model = tf.keras.models.load_model(severity_model_path)
    return classification_model, severity_model

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=9.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return enhanced_image

def gamma_correction(image, gamma=1.5):
    image = image.astype(np.float32) / 255.0
    corrected_image = np.power(image, gamma)
    corrected_image = np.clip(corrected_image * 255, 0, 255).astype(np.uint8)
    return corrected_image

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_clahe = apply_clahe(image)
    image_corrected = gamma_correction(image_clahe, gamma=1.5)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_corrected)
    plt.axis("off")
    plt.show()

    final_image = cv2.resize(image_corrected, (224, 224))
    final_image = final_image / 255.0
    final_image = np.expand_dims(final_image, axis=0)
    
    return final_image

def classify_retinal_disease(image_path, classification_model, severity_model):
    image = preprocess_image(image_path)  

    prediction = classification_model.predict(image)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    confidence = prediction[0][predicted_class]

    print("\n🔹Diagnosis Report🔹")
    print(f"Detected Disease: {predicted_label} (Confidence: {confidence:.2f})")

    if predicted_label == "DR":
        severity_prediction = severity_model.predict(image)
        severity_class = np.argmax(severity_prediction)
        severity_label = severity_labels[severity_class]
    else:
        severity_label = "Mild" if confidence > 0.8 else "Moderate" if confidence > 0.5 else "Severe"

    print(f"Severity Level: {severity_label}")
    print(f"Description: {disease_info[predicted_label]['description']}")
    print(f"Recommended Diagnosis: {disease_info[predicted_label]['diagnosis']}")

    return predicted_label, severity_label

classification_model_path = r"C:\Users\Yuktha Varshika\Music\Models\Retinal_Disease_Classification.h5"
severity_model_path = r"C:\Users\Yuktha Varshika\Music\Models\Retinal_Severity_Classification.h5"

classification_model, severity_model = load_models(classification_model_path, severity_model_path)

test_image_path = r"C:\Users\Yuktha Varshika\Music\Classified\Hypertensive Retinopathy\00000ce7.png"

classify_retinal_disease(test_image_path, classification_model, severity_model)
