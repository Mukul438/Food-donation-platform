# app.py (updated)
import os
import time
import logging
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# ---------- Config ----------
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
app.secret_key = "food_waste_secret"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///food_waste.db"
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 4 MB max upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

db = SQLAlchemy(app)

# ---------- Database Models ----------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    alerts = db.relationship('FoodAlert', backref='owner', lazy=True)

class FoodAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(200), nullable=False)
    quantity = db.Column(db.String(50), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    date_posted = db.Column(db.DateTime, default=datetime.utcnow)
    collected = db.Column(db.Boolean, default=False)
    image_filename = db.Column(db.String(200))  # image file name (stored in static/uploads)
    prediction = db.Column(db.String(100))      # AI model result
    posted_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# ---------- Helper functions ----------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- Load AI model ----------
# IMPORTANT: Ensure these labels match the class order used when training your model.
# From your training script earlier the order was: ['cooked_food','fruits','others','vegetables']
# Update below if your model uses a different order.
class_labels = ["cooked_food", "fruits", "others", "vegetables"]

try:
    model = tf.keras.models.load_model("food_waste_model.h5")
    logging.info("✅ Loaded AI model: food_waste_model.h5")
except Exception as e:
    model = None
    logging.error(f"Failed to load model: {e}")

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

# Signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        role = request.form['role']

        if User.query.filter_by(username=username).first():
            flash("⚠️ User already exists!", "danger")
            return redirect(url_for("signup"))

        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
        new_user = User(username=username, password=hashed_password, role=role)
        db.session.add(new_user)
        db.session.commit()
        flash("✅ Account created successfully!", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')

# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["role"] = user.role
            session["username"] = user.username
            return redirect(url_for("mess_dashboard" if user.role == "mess" else "ngo_dashboard"))
        else:
            flash("❌ Invalid username or password.", "danger")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# Mess Dashboard
@app.route("/mess_dashboard", methods=["GET", "POST"])
def mess_dashboard():
    if "role" not in session or session["role"] != "mess":
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            flash("⚠️ Food image is required for AI prediction!", "danger")
            return redirect(url_for("mess_dashboard"))

        # Save image
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(image_path)

        # AI Prediction
        img = image.load_img(image_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        predicted_class = class_labels[np.argmax(preds)]

        # Create food alert WITH AI result
        alert = FoodAlert(
            description=request.form["description"],
            quantity=request.form["quantity"],
            location=request.form["location"],
            image_filename=filename,
            prediction=predicted_class,
            posted_by=session["user_id"]
        )

        db.session.add(alert)
        db.session.commit()

        flash(f"✅ Food posted with AI category: {predicted_class}", "success")
        return redirect(url_for("mess_dashboard"))

    alerts = FoodAlert.query.filter_by(posted_by=session["user_id"]).all()
    return render_template("mess_dashboard.html", alerts=alerts)


# NGO Dashboard
@app.route("/ngo_dashboard")
def ngo_dashboard():
    if "role" not in session or session["role"] != "ngo":
        return redirect(url_for("login"))

    alerts = FoodAlert.query.filter_by(collected=False).all()
    return render_template("ngo_dashboard.html", alerts=alerts)

# Collect alert (NGO action) - the template posts to this route
@app.route("/collect/<int:alert_id>", methods=["POST"])
def collect_alert(alert_id):
    if "role" not in session or session["role"] != "ngo":
        return redirect(url_for("login"))

    alert = FoodAlert.query.get(alert_id)
    if alert:
        alert.collected = True
        db.session.commit()
        flash("✅ Food collected successfully!", "success")
    return redirect(url_for("ngo_dashboard"))

# Mark collected (mess owner) and delete
@app.route('/mark_collected/<int:alert_id>')
def mark_collected(alert_id):
    alert = FoodAlert.query.get_or_404(alert_id)
    alert.collected = True
    db.session.commit()
    flash("✅ Food marked as collected!", "success")
    return redirect(url_for('mess_dashboard'))

@app.route('/delete_alert/<int:alert_id>')
def delete_alert(alert_id):
    alert = FoodAlert.query.get_or_404(alert_id)
    # optionally remove the image file too
    if alert.image_filename:
        try:
            os.remove(os.path.join(app.config["UPLOAD_FOLDER"], alert.image_filename))
        except Exception:
            pass
    db.session.delete(alert)
    db.session.commit()
    flash("🗑️ Food post deleted!", "danger")
    return redirect(url_for('mess_dashboard'))

# ---------- AI Image Classifier ----------
@app.route("/ai_classifier", methods=["GET", "POST"])
def ai_classifier():
    # allow both roles to use classifier; change to only 'ngo' if desired
    if "role" not in session or session["role"] not in ("ngo", "mess"):
        return redirect(url_for("login"))

    prediction = None
    image_path = None
    saved_filename = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("⚠️ Please upload an image!", "danger")
            return redirect(url_for("ai_classifier"))

        if not allowed_file(file.filename):
            flash("⚠️ Invalid file type. Use png/jpg/jpeg/gif.", "danger")
            return redirect(url_for("ai_classifier"))

        # Secure and unique filename
        orig_name = secure_filename(file.filename)
        saved_filename = f"{int(time.time())}_{orig_name}"
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)

        try:
            file.save(image_path)

            # Preprocess and predict
            if model is None:
                flash("⚠️ AI model not available on server.", "danger")
                # remove saved image if model missing
                try:
                    os.remove(image_path)
                except:
                    pass
                return redirect(url_for("ai_classifier"))

            img = image.load_img(image_path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred_probs = model.predict(img_array)
            predicted_class = class_labels[int(np.argmax(pred_probs))]
            prediction = predicted_class

            # Save prediction info into FoodAlert (optional fields)
            # You can decide quantity/location or ask user to fill later; using placeholders here
            new_alert = FoodAlert(
                description=f"AI: {predicted_class}",
                quantity="Unknown",
                location="Not specified",
                image_filename=saved_filename,
                prediction=predicted_class,
                posted_by=session["user_id"]
            )
            db.session.add(new_alert)
            db.session.commit()

            flash(f"✅ Prediction: {predicted_class}", "success")
        except Exception as e:
            logging.error(f"AI prediction error: {e}", exc_info=True)
            flash("⚠️ Error processing image. Try another file.", "danger")
            try:
                if saved_filename:
                    os.remove(os.path.join(app.config["UPLOAD_FOLDER"], saved_filename))
            except:
                pass
            return redirect(url_for("ai_classifier"))

    # For rendering, construct path expected by template (static/uploads/<filename>)
    if image_path:
        # only pass filename (template constructs url_for)
        template_image_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
    else:
        template_image_path = None

    return render_template("ai_classifier.html", prediction=prediction, image_path=template_image_path)

# ---------- Run ----------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
