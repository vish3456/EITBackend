from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from dotenv import load_dotenv
from datetime import datetime

# 1. Load environment variables
load_dotenv()

app = Flask(__name__)

# 2. Updated CORS to be more robust for deployment
CORS(app, resources={r"/*": {"origins": "*"}})

# 3. Database Configuration
# We will use aashray.db as your primary name
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///aashray.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ── DATABASE MODELS ──────────────────────────────────────────────────
class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(50), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class AssessmentScore(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    composite = db.Column(db.Integer, nullable=False)
    stress = db.Column(db.Integer)
    anxiety = db.Column(db.Integer)
    risk_level = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ─── CRITICAL FIX: FORCED INITIALIZATION FOR PRODUCTION (GUNICORN) ───
with app.app_context():
    try:
        db.create_all()
        print("✅ Database successfully initialized and tables created.")
    except Exception as e:
        print(f"❌ Initial DB Error: {str(e)}")
# ─────────────────────────────────────────────────────────────────────

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "Aashray Backend & Database is Running! 🚀"
    })

@app.route('/api/gemini', methods=['POST'])
def call_gemini():
    data = request.json
    messages = data.get('messages', [])
    system_prompt = data.get('system', '')
    is_chat = "You are Aasha" in system_prompt

    try:
        if is_chat and messages:
            latest_user_msg = messages[-1]['content']
            db.session.add(ChatMessage(role='user', content=latest_user_msg))
            db.session.commit()

        gemini_history = []
        for msg in messages:
            role = "model" if msg['role'] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [msg['content']]})

        if gemini_history and gemini_history[0]['role'] == 'model':
            gemini_history.pop(0)

        if not gemini_history:
            return jsonify({"error": "No valid user messages"}), 400

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_prompt
        )

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        latest_prompt = gemini_history.pop()

        if is_chat:
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(latest_prompt['parts'][0], safety_settings=safety_settings)
        else:
            response = model.generate_content(latest_prompt['parts'][0], safety_settings=safety_settings)

        reply_text = response.text

        if is_chat:
            db.session.add(ChatMessage(role='model', content=reply_text))
            db.session.commit()

        return jsonify({"reply": reply_text})

    except Exception as e:
        print(f"Gemini Error: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/save_assessment', methods=['POST'])
def save_assessment():
    data = request.json
    try:
        new_score = AssessmentScore(
            composite=data.get('composite', 0),
            stress=data.get('stress', 0),
            anxiety=data.get('anxiety', 0),
            risk_level=data.get('risk', 'UNKNOWN')
        )
        db.session.add(new_score)
        db.session.commit()
        return jsonify({"status": "success", "message": "Assessment saved!"})
    except Exception as e:
        print(f"Database Error: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
