from server import app, db, User

with app.app_context():
    # Create all tables
    db.create_all()

    # Add demo users
    if not User.query.filter_by(username="mess_owner").first():
        mess_owner = User(username="mess_owner", password="1234", role="owner")
        ngo = User(username="ngo_user", password="1234", role="ngo")
        db.session.add(mess_owner)
        db.session.add(ngo)
        db.session.commit()
        print("Demo users created!")
    else:
        print("Users already exist!")
