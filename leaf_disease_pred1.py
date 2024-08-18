from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import google.generativeai as genai
import os

app = Flask(__name__)

# Load your trained model
model = load_model(r"C:\Users\Venumuddala Ashritha\plant_disease_prediction_model.h5")

# Define the label mapping
labels = {0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust', 3: 'Apple___healthy',
          4: 'Blueberry___healthy', 5: 'Cherry_(including_sour)___Powdery_mildew', 6: 'Cherry_(including_sour)___healthy',
          7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 8: 'Corn_(maize)___Common_rust_',
          9: 'Corn_(maize)___Northern_Leaf_Blight', 10: 'Corn_(maize)___healthy', 11: 'Grape___Black_rot',
          12: 'Grape___Esca_(Black_Measles)', 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 14: 'Grape___healthy',
          15: 'Orange___Haunglongbing_(Citrus_greening)', 16: 'Peach___Bacterial_spot', 17: 'Peach___healthy',
          18: 'Pepper,_bell___Bacterial_spot', 19: 'Pepper,_bell___healthy', 20: 'Potato___Early_blight',
          21: 'Potato___Late_blight', 22: 'Potato___healthy', 23: 'Raspberry___healthy', 24: 'Soybean___healthy',
          25: 'Squash___Powdery_mildew', 26: 'Strawberry___Leaf_scorch', 27: 'Strawberry___healthy',
          28: 'Tomato___Bacterial_spot', 29: 'Tomato___Early_blight', 30: 'Tomato___Late_blight', 31: 'Tomato___Leaf_Mold',
          32: 'Tomato___Septoria_leaf_spot', 33: 'Tomato___Spider_mites Two-spotted_spider_mite', 34: 'Tomato___Target_Spot',
          35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 36: 'Tomato___Tomato_mosaic_virus', 37: 'Tomato___healthy'}

# Function to process image and predict the label
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust based on your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    predictions = model.predict(img_array)
    predicted_label = labels[np.argmax(predictions[0])]
    return predicted_label

def format_gemini_output(text):
    formatted_text = ""

    # Split the text into parts by "**" to detect headings and by "*" to detect list items
    parts = text.split("**")
    
    for part in parts:
        if part.strip():
            # If part contains a heading, it will not contain "*", so we handle it separately
            if "*" not in part:
                formatted_text += f"\n\n{part.strip()}\n"
            else:
                # Split by "*" to get the list items
                sub_parts = part.split("*")
                for sub_part in sub_parts:
                    if sub_part.strip():
                        formatted_text += f"- {sub_part.strip()}\n"

    return formatted_text.strip()
# Function to get analysis from Gemini API
def get_gemini_analysis(text):
    # Initialize the model
    genai.configure(api_key="AIzaSyDz9Q--Wq4MaLVTMywxieiRjpuMkYi-uk4")
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Generate content
    response = model.generate_content(text)
    return response.text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(r"C:\Users\Venumuddala Ashritha\Downloads\plant_dis_pred\plantvillage dataset", file.filename)
            file.save(file_path)
            predicted_label = predict_disease(file_path)

            # Get Gemini analysis
            gemini_text1 = f"In very simple 2 to 3 points, give details of plant disease: {predicted_label}"
            gemini_analysis1 = get_gemini_analysis(gemini_text1)

            gemini_text2 = f"In very simple 2 to 3 points, give effective ways of what to be done by farmer for plant disease: {predicted_label}"
            gemini_analysis2 = get_gemini_analysis(gemini_text2)

            gemini_text3 = f"In very simple 2 to 3 points, suggest the organic ways and organic fertilizers to the farmer for plant disease: {predicted_label}"
            gemini_analysis3 = get_gemini_analysis(gemini_text3)

            gemini_analysis_final1 = format_gemini_output(gemini_analysis1)
            gemini_analysis_final2 = format_gemini_output(gemini_analysis2)
            gemini_analysis_final3 = format_gemini_output(gemini_analysis3)

            return render_template(
                'result11.html', 
                prediction=predicted_label, 
                details=gemini_analysis_final1,
                effective_ways=gemini_analysis_final2,
                organic_suggestions=gemini_analysis_final3,
                image_path=file_path
            )
    
    return render_template('index11.html')

if __name__ == '__main__':
    app.run(debug=True)
