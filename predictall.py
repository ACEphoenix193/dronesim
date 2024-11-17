import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import random

def load_model(model_path):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)
    return model

def get_class_names(dataset_dir):
   
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    return class_names

def predict(model, img_path, class_names, image_size=(256, 256)): 
    
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  
    img_array = img_array / 255.0 

    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    return class_names[predicted_class[0]]

def get_random_image_from_class(class_dir):
   
    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
  
    random_image = random.choice(image_files)
    
    return os.path.join(class_dir, image_files)

if __name__ == "__main__":
    model_path = "model.h5"   
    dataset_dir = "data/valid"  
    class_names = get_class_names(dataset_dir)
    print(f"Class names: {class_names}")
    
    model = load_model(model_path)
    
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        random_image_path = get_random_image_from_class(class_dir)
        
        
        predicted_class = predict(model, random_image_path, class_names)
        
        # Print the results
        print(f"Class: {class_name}, Image: {random_image_path}, Predicted: {predicted_class}")
