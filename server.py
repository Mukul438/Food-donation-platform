from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "food_waste_secret"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///food_waste.db"
db = SQLAlchemy(app)

# ----------------------
# Database Models
# ----------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)  # hashed password
    role = db.Column(db.String(20), nullable=False)       # "mess" or "ngo"
    alerts = db.relationship('FoodAlert', backref='owner', lazy=True)


class FoodAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(200), nullable=False)
    quantity = db.Column(db.String(50), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    date_posted = db.Column(db.DateTime, default=datetime.utcnow)
    collected = db.Column(db.Boolean, default=False)

    posted_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# ----------------------
# Routes
# ----------------------
@app.route("/")
def index():
    return render_template("index.html")


# ----------------------
# Signup Route
# ----------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']

        # check if user already exists
        if User.query.filter_by(username=username).first():
            flash("⚠️ User already exists!", "danger")
            return redirect(url_for("signup"))

        # create new user with hashed password
        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
        new_user = User(username=username, password=hashed_password, role=role)
        db.session.add(new_user)
        db.session.commit()
        flash("✅ Account created successfully! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')


# ----------------------
# Login Route
# ----------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["role"] = user.role
            session["username"] = user.username
            if user.role == "mess":
                return redirect(url_for("mess_dashboard"))
            else:
                return redirect(url_for("ngo_dashboard"))
        else:
            error = "❌ Invalid username or password. Please try again."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ----------------------
# Mess Owner Dashboard
# ----------------------
@app.route("/mess_dashboard", methods=["GET", "POST"])
def mess_dashboard():
    if "role" not in session or session["role"] != "mess":
        return redirect(url_for("login"))

    if request.method == "POST":
        description = request.form["description"]
        quantity = request.form["quantity"]
        location = request.form["location"]

        alert = FoodAlert(description=description,
                          quantity=quantity,
                          location=location,
                          posted_by=session["user_id"])
        db.session.add(alert)
        db.session.commit()
        flash("✅ Food post added!", "success")
        return redirect(url_for("mess_dashboard"))

    alerts = FoodAlert.query.filter_by(posted_by=session["user_id"]).all()
    return render_template("mess_dashboard.html", alerts=alerts)


# ----------------------
# NGO Dashboard
# ----------------------
@app.route("/ngo_dashboard")
def ngo_dashboard():
    if "role" not in session or session["role"] != "ngo":
        return redirect(url_for("login"))

    alerts = FoodAlert.query.filter_by(collected=False).all()
    return render_template("ngo_dashboard.html", alerts=alerts)


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
    db.session.delete(alert)
    db.session.commit()
    flash("🗑️ Food post deleted!", "danger")
    return redirect(url_for('mess_dashboard'))


# ----------------------
# Run Server
# ----------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create tables if not exist
    app.run(debug=True)
