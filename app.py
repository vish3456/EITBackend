import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# 1. SETUP & CONFIGURATION
load_dotenv()
app = Flask(__name__)

# Global CORS: Allows Vercel/GitHub Pages to communicate with Render
CORS(app, resources={r"/*": {"origins": "*"}})

# Database path (SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///aashray.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 2. DATABASE MODELS
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

# 3. AUTO-INITIALIZE TABLES (Essential for Render)
with app.app_context():
    try:
        db.create_all()
        print("✅ Database successfully initialized and tables verified.")
    except Exception as e:
        print(f"❌ Initial DB Error: {str(e)}")

# 4. AI CONFIGURATION
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not found in environment variables!")
genai.configure(api_key=API_KEY)

# 5. ROUTES
@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "Aashray Backend & Database is Running! 🚀",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/api/gemini', methods=['POST'])
def call_gemini():
    data = request.json or {}
    messages = data.get('messages', [])
    system_prompt = data.get('system', 'You are Aasha, a trauma-informed assistant.')
    is_chat = "You are Aasha" in system_prompt

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    try:
        # DB Log: User Message
        if is_chat:
            db.session.add(ChatMessage(role='user', content=messages[-1]['content']))
            db.session.commit()

        # Format history for Gemini API
        gemini_history = []
        for msg in messages[:-1]:  # All but the last message
            role = "model" if msg['role'] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [msg['content']]})

        # Model Setup (Stable 1.5 Flash)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_prompt
        )

        # Safety Settings (Ensures support content isn't blocked)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Generate Response
        last_msg_text = messages[-1]['content']
        
        if is_chat:
            chat_session = model.start_chat(history=gemini_history)
            response = chat_session.send_message(last_msg_text, safety_settings=safety_settings)
        else:
            # For Assessment Scoring
            response = model.generate_content(last_msg_text, safety_settings=safety_settings)

        reply_text = response.text

        # DB Log: AI Message
        if is_chat:
            db.session.add(ChatMessage(role='model', content=reply_text))
            db.session.commit()

        return jsonify({"reply": reply_text})

    except Exception as e:
        db.session.rollback()
        error_str = str(e)
        print(f"🔴 Backend Error: {error_str}")
        
        # Friendly response for quota errors
        if "429" in error_str:
            return jsonify({"error": "AI Quota exceeded. Using local fallback.", "code": 429}), 429
        return jsonify({"error": error_str}), 500

@app.route('/api/save_assessment', methods=['POST'])
def save_assessment():
    data = request.json or {}
    try:
        new_score = AssessmentScore(
            composite=data.get('composite', 0),
            stress=data.get('stress', 0),
            anxiety=data.get('anxiety', 0),
            risk_level=data.get('risk', 'UNKNOWN')
        )
        db.session.add(new_score)
        db.session.commit()
        return jsonify({"status": "success", "message": "Assessment logged to SQLite!"})
    except Exception as e:
        db.session.rollback()
        print(f"🔴 DB Error: {e}")
        return jsonify({"error": "Failed to save to database"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
