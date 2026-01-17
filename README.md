# 🍽️ Food Waste Management System (AI Powered)

An AI-powered full-stack web application that reduces food waste by connecting mess owners with NGOs. 
The platform uses deep learning–based image classification and pickup-time prioritization to enable efficient food redistribution.

---

## 🚀 Features
- Role-based authentication (Mess Owner / NGO)
- AI-powered food image classification (TensorFlow)
- Pickup time & urgency handling
- NGO dashboard with filters & priority badges
- Food collection tracking
- Analytics dashboard

---

## 🛠 Tech Stack
- **Backend:** Python, Flask, SQLAlchemy
- **Frontend:** HTML, CSS, Bootstrap
- **Database:** SQLite
- **AI/ML:** TensorFlow, Keras, CNN
- **Tools:** Git, GitHub, VS Code

---

## 🔮 Future Enhancements
- Live notifications for NGOs
- Cloud deployment
- Mobile-friendly UI
- Email/WhatsApp alerts

---

## 📦 Large Files Note

Due to GitHub file size limitations, the trained model file (`food_waste_model.h5`) and dataset are not included in this repository.

- The dataset was used for training a CNN-based image classification model.
- The model can be retrained using `train_model.py`.
- Users can generate the model locally by running the training script.

This approach follows standard practices for managing large ML assets.

## ⚙️ How to Run Locally

```bash
git clone https://github.com/your-username/food-waste-ai-project.git
cd food-waste-ai-project
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
python app.py

 
