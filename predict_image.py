import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained model
model = load_model('food_waste_model.h5')

# List of classes in order
classes = ['cooked_food', 'fruits', 'others', 'vegetables']

# Path of the test image
img_path =r'C:\Users\mukul\Downloads\food_waste_ai_project\dataset\train\fruits\apple.jpg'
  # Change this to your image file name
img = image.load_img(img_path, target_size=(128, 128))

# Convert image to array and normalize
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
predictions = model.predict(img_array)
predicted_class = classes[np.argmax(predictions)]

print(f"Prediction: {predicted_class}")
