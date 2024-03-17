from flask_sqlalchemy import SQLAlchemy
from flask import Flask
import os

app = Flask(__name__)

# Will be updated soon
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define the Player model
class Player(db.Model):

    # Will be updated to have all the columns
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(60), unique=True, nullable=False)
    team = db.Column(db.String(40), nullable=False)
    position = db.Column(db.String(3), nullable=False)

    def __repr__(self):
        return '<Player %r>' % self.name

# Would be used if I want to call the initialization from app.py
def init_db():
    with app.app_context():
        db.create_all()

@app.before_first_request
def create_tables():
    db.create_all()
