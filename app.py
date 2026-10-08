# app.py

import os
import time
import uuid
import logging
from datetime import datetime

import numpy as np
import tensorflow.lite as tflite

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)

from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.preprocessing import image


# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

app.secret_key = "food_waste_secret"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///food_waste.db"
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

db = SQLAlchemy(app)


# ============================================================
# DATABASE MODELS
# ============================================================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)

    alerts = db.relationship(
        "FoodAlert",
        backref="owner",
        lazy=True
    )


class FoodAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    description = db.Column(
        db.String(200),
        nullable=False
    )

    quantity = db.Column(
        db.String(50),
        nullable=False
    )

    location = db.Column(
        db.String(100),
        nullable=False
    )

    date_posted = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    collected = db.Column(
        db.Boolean,
        default=False
    )

    image_filename = db.Column(
        db.String(200)
    )

    prediction = db.Column(
        db.String(100)
    )

    posted_by = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# AI MODEL
# ============================================================

class_labels = [
    "cooked_food",
    "fruits",
    "others",
    "vegetables"
]


# ---------- Load TFLite model ----------

try:

    tflite_model = tflite.Interpreter(
        model_path="food_waste_model.tflite"
    )

    tflite_model.allocate_tensors()

    tflite_input_details = (
        tflite_model.get_input_details()
    )

    tflite_output_details = (
        tflite_model.get_output_details()
    )

    logging.info(
        "✅ Loaded TFLite AI model: food_waste_model.tflite"
    )

except Exception as e:

    tflite_model = None
    tflite_input_details = None
    tflite_output_details = None

    logging.error(
        f"Failed to load TFLite model: {e}"
    )


# ---------- AI prediction helper ----------

def predict_food(image_path):

    if tflite_model is None:
        raise RuntimeError(
            "TFLite AI model is not available."
        )

    # Load image
    img = image.load_img(
        image_path,
        target_size=(128, 128)
    )

    # Convert image to array
    img_array = image.img_to_array(img)

    # Normalize exactly like the original model
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Get expected input type
    input_dtype = (
        tflite_input_details[0]["dtype"]
    )

    img_array = img_array.astype(
        input_dtype
    )

    # Set input
    tflite_model.set_tensor(
        tflite_input_details[0]["index"],
        img_array
    )

    # Run model
    tflite_model.invoke()

    # Get prediction
    predictions = tflite_model.get_tensor(
        tflite_output_details[0]["index"]
    )

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_labels[
        predicted_index
    ]

    return predicted_class, predictions


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# SIGNUP
# ============================================================

@app.route(
    "/signup",
    methods=["GET", "POST"]
)
def signup():

    if request.method == "POST":

        username = request.form[
            "username"
        ].strip()

        password = request.form[
            "password"
        ]

        role = request.form[
            "role"
        ]

        # Check existing user
        if User.query.filter_by(
            username=username
        ).first():

            flash(
                "⚠️ User already exists!",
                "danger"
            )

            return redirect(
                url_for("signup")
            )

        # Hash password
        hashed_password = generate_password_hash(
            password,
            method="pbkdf2:sha256"
        )

        new_user = User(
            username=username,
            password=hashed_password,
            role=role
        )

        db.session.add(
            new_user
        )

        db.session.commit()

        flash(
            "✅ Account created successfully!",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "signup.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = request.form[
            "username"
        ].strip()

        password = request.form[
            "password"
        ]

        user = User.query.filter_by(
            username=username
        ).first()

        if user and check_password_hash(
            user.password,
            password
        ):

            session["user_id"] = user.id
            session["role"] = user.role
            session["username"] = user.username

            if user.role == "mess":

                return redirect(
                    url_for("mess_dashboard")
                )

            else:

                return redirect(
                    url_for("ngo_dashboard")
                )

        else:

            flash(
                "❌ Invalid username or password.",
                "danger"
            )

    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("index")
    )


# ============================================================
# MESS DASHBOARD
# ============================================================

@app.route(
    "/mess_dashboard",
    methods=["GET", "POST"]
)
def mess_dashboard():

    # Only mess users
    if (
        "role" not in session
        or session["role"] != "mess"
    ):

        return redirect(
            url_for("login")
        )

    # ---------------- POST ----------------

    if request.method == "POST":

        logging.info(
            "UPLOAD: Request received"
        )

        file = request.files.get(
            "image"
        )

        # Check image
        if (
            not file
            or file.filename == ""
        ):

            flash(
                "Please select a food image.",
                "danger"
            )

            return redirect(
                url_for("mess_dashboard")
            )

        # Check extension
        if not allowed_file(
            file.filename
        ):

            flash(
                "Invalid image format. "
                "Use PNG, JPG, JPEG or GIF.",
                "danger"
            )

            return redirect(
                url_for("mess_dashboard")
            )

        # Check AI model
        if tflite_model is None:

            logging.error(
                "UPLOAD: TFLite AI model unavailable"
            )

            flash(
                "AI model is not available.",
                "danger"
            )

            return redirect(
                url_for("mess_dashboard")
            )

        # Unique filename
        original_filename = secure_filename(
            file.filename
        )

        filename = (
            f"{uuid.uuid4().hex}_"
            f"{original_filename}"
        )

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        try:

            # ---------------- Save image ----------------

            logging.info(
                "UPLOAD: Saving image"
            )

            file.save(
                image_path
            )

            logging.info(
                "UPLOAD: Image saved. "
                "Preparing image"
            )

            # ---------------- AI prediction ----------------

            logging.info(
                "UPLOAD: Starting TFLite prediction"
            )

            predicted_class, predictions = (
                predict_food(image_path)
            )

            logging.info(
                "UPLOAD: Prediction completed"
            )

            logging.info(
                f"UPLOAD: Predicted class = "
                f"{predicted_class}"
            )

            # ---------------- Save food alert ----------------

            alert = FoodAlert(

                description=request.form.get(
                    "description",
                    ""
                ),

                quantity=request.form.get(
                    "quantity",
                    ""
                ),

                location=request.form.get(
                    "location",
                    ""
                ),

                image_filename=filename,

                prediction=predicted_class,

                posted_by=session[
                    "user_id"
                ]
            )

            db.session.add(
                alert
            )

            db.session.commit()

            logging.info(
                "UPLOAD: Food alert saved successfully"
            )

            flash(
                f"Food posted successfully! "
                f"AI category: {predicted_class}",
                "success"
            )

        except Exception:

            db.session.rollback()

            logging.exception(
                "UPLOAD: Image processing failed"
            )

            # Delete uploaded image
            if os.path.exists(
                image_path
            ):

                try:
                    os.remove(
                        image_path
                    )

                except Exception:
                    pass

            flash(
                "Unable to process image. "
                "Please try again.",
                "danger"
            )

        return redirect(
            url_for("mess_dashboard")
        )

    # ---------------- GET ----------------

    alerts = FoodAlert.query.filter_by(
        posted_by=session["user_id"]
    ).all()

    return render_template(
        "mess_dashboard.html",
        alerts=alerts
    )


# ============================================================
# NGO DASHBOARD
# ============================================================

@app.route("/ngo_dashboard")
def ngo_dashboard():

    if (
        "role" not in session
        or session["role"] != "ngo"
    ):

        return redirect(
            url_for("login")
        )

    alerts = FoodAlert.query.filter_by(
        collected=False
    ).all()

    return render_template(
        "ngo_dashboard.html",
        alerts=alerts
    )


# ============================================================
# COLLECT FOOD - NGO
# ============================================================

@app.route(
    "/collect/<int:alert_id>",
    methods=["POST"]
)
def collect_alert(alert_id):

    if (
        "role" not in session
        or session["role"] != "ngo"
    ):

        return redirect(
            url_for("login")
        )

    alert = FoodAlert.query.get(
        alert_id
    )

    if alert:

        alert.collected = True

        db.session.commit()

        flash(
            "✅ Food collected successfully!",
            "success"
        )

    return redirect(
        url_for("ngo_dashboard")
    )


# ============================================================
# MARK COLLECTED - MESS
# ============================================================

@app.route(
    "/mark_collected/<int:alert_id>"
)
def mark_collected(alert_id):

    alert = FoodAlert.query.get_or_404(
        alert_id
    )

    alert.collected = True

    db.session.commit()

    flash(
        "✅ Food marked as collected!",
        "success"
    )

    return redirect(
        url_for("mess_dashboard")
    )


# ============================================================
# DELETE FOOD ALERT
# ============================================================

@app.route(
    "/delete_alert/<int:alert_id>"
)
def delete_alert(alert_id):

    alert = FoodAlert.query.get_or_404(
        alert_id
    )

    # Remove image file
    if alert.image_filename:

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            alert.image_filename
        )

        try:

            if os.path.exists(
                image_path
            ):

                os.remove(
                    image_path
                )

        except Exception:

            pass

    # Delete database record
    db.session.delete(
        alert
    )

    db.session.commit()

    flash(
        "🗑️ Food post deleted!",
        "danger"
    )

    return redirect(
        url_for("mess_dashboard")
    )


# ============================================================
# AI IMAGE CLASSIFIER
# ============================================================

@app.route(
    "/ai_classifier",
    methods=["GET", "POST"]
)
def ai_classifier():

    # Both roles can use classifier
    if (
        "role" not in session
        or session["role"]
        not in ("ngo", "mess")
    ):

        return redirect(
            url_for("login")
        )

    prediction = None
    image_path = None
    saved_filename = None

    # ---------------- POST ----------------

    if request.method == "POST":

        file = request.files.get(
            "image"
        )

        if (
            not file
            or file.filename == ""
        ):

            flash(
                "⚠️ Please upload an image!",
                "danger"
            )

            return redirect(
                url_for("ai_classifier")
            )

        if not allowed_file(
            file.filename
        ):

            flash(
                "⚠️ Invalid file type. "
                "Use png/jpg/jpeg/gif.",
                "danger"
            )

            return redirect(
                url_for("ai_classifier")
            )

        if tflite_model is None:

            flash(
                "⚠️ AI model not available on server.",
                "danger"
            )

            return redirect(
                url_for("ai_classifier")
            )

        # Secure unique filename
        orig_name = secure_filename(
            file.filename
        )

        saved_filename = (
            f"{uuid.uuid4().hex}_"
            f"{orig_name}"
        )

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            saved_filename
        )

        try:

            # Save image
            file.save(
                image_path
            )

            # TFLite prediction
            predicted_class, predictions = (
                predict_food(image_path)
            )

            prediction = predicted_class

            # Save prediction as FoodAlert
            new_alert = FoodAlert(

                description=(
                    f"AI: {predicted_class}"
                ),

                quantity="Unknown",

                location="Not specified",

                image_filename=saved_filename,

                prediction=predicted_class,

                posted_by=session[
                    "user_id"
                ]
            )

            db.session.add(
                new_alert
            )

            db.session.commit()

            flash(
                f"✅ Prediction: "
                f"{predicted_class}",
                "success"
            )

        except Exception:

            db.session.rollback()

            logging.exception(
                "AI prediction error"
            )

            if (
                saved_filename
                and os.path.exists(
                    os.path.join(
                        app.config["UPLOAD_FOLDER"],
                        saved_filename
                    )
                )
            ):

                try:

                    os.remove(
                        os.path.join(
                            app.config["UPLOAD_FOLDER"],
                            saved_filename
                        )
                    )

                except Exception:

                    pass

            flash(
                "⚠️ Error processing image. "
                "Try another file.",
                "danger"
            )

            return redirect(
                url_for("ai_classifier")
            )

    # ---------------- Render page ----------------

    if image_path:

        template_image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            saved_filename
        )

    else:

        template_image_path = None

    return render_template(
        "ai_classifier.html",
        prediction=prediction,
        image_path=template_image_path
    )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

with app.app_context():

    db.create_all()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )