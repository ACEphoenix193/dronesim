import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

def load_model(model_path="plant_disease_model2.h5"):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)
    return model

def get_class_names(dataset_dir):
    
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    return class_names

def predict(model, img_path, class_names, image_size=(256, 256)): 
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  
    img_array = img_array / 255.0  
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    
    return class_names[predicted_class[0]]

if __name__ == "__main__":
    
    model_path = "model.h5"
    img_path = input("Enter path to image: ")  # Replace with your image path
    dataset_dir = "data/valid"  

    
    class_names = get_class_names(dataset_dir)
    print(f"Class names: {class_names}")

    
    model = load_model(model_path)

   
    result = predict(model, img_path, class_names)
    print(f"Predicted Class: {result}")
