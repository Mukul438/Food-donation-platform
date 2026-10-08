# Food Waste Management System

A web application that helps connect messes with NGOs so that surplus food can be shared instead of being wasted.

The application allows mess users to post available food along with its quantity, location, and image. NGOs can view the available food and mark it as collected. A food image classification model is also included to identify the type of food from an uploaded image.

## Live Demo

https://food-donation-platform-0b5c.onrender.com

## Features

- User registration and login
- Separate dashboards for Mess and NGO users
- Mess users can post surplus food
- Upload food images
- Food image classification
- Add food quantity and location
- NGOs can view available food
- NGOs can collect food donations
- Track food collection status
- Delete food posts
- SQLite database for storing application data

## Food Classification

The application includes an image classification model that categorizes food images into four classes:

- Cooked Food
- Fruits
- Others
- Vegetables

The original model was trained using TensorFlow/Keras and converted to TensorFlow Lite for use in the web application.

## Technologies Used

- Python
- Flask
- Flask-SQLAlchemy
- TensorFlow
- TensorFlow Lite
- NumPy
- Pillow
- SQLite
- HTML
- CSS
- Bootstrap
- JavaScript
- Git & GitHub
- Render

## How It Works

1. A mess user creates an account and logs in.
2. The user enters the food details such as description, quantity, and location.
3. A food image is uploaded.
4. The image is processed by the classification model.
5. The food post is saved in the database.
6. NGOs can view the available food posts.
7. An NGO can collect a food post.
8. The status of the food post is updated after collection.

## Project Structure

```text
Food-donation-platform/
│
├── app.py
├── food_waste_model.tflite
├── food_waste_model.h5
├── requirements.txt
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── mess_dashboard.html
│   ├── ngo_dashboard.html
│   └── ai_classifier.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
│
└── README.md
