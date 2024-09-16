from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained CNN model
model = load_model("cnn_model_final_2.h5")

# Define the function to process the image
def prepare_image(image, target_size):
    image = load_img(image, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # Load the model using the factory pattern
    #model = ModelFactory.get_model()

    # Check if the file is in the request
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # Get the file from the request
    file = request.files["file"]

    # Convert the file to a PIL Image
    img = Image.open(BytesIO(file.read()))
    
    # Resize the image to match the model's input size
    img = img.resize((320, 320))

    # Convert the image to an array and expand dimensions to match the input shape expected by the model
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Make the prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])

    # Map the predicted class to freshwater fish species
    species_dict = {0: "Angel_fish", 1: "Asian_Arowana", 2:"Cardinal_Tetra", 3:"Clown_Loach"} # species
    fish_species = species_dict.get(predicted_class, "Unknown")
    print(fish_species)
    return jsonify({"prediction": fish_species})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
