from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# ── DATABASE CONFIGURATION ───────────────────────────────────────────────
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///safeher.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


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


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


@app.route('/')
def home():
    return "Aashray Backend & Database is Running! 🚀"


@app.route('/api/gemini', methods=['POST'])
def call_gemini():
    data = request.json
    messages = data.get('messages', [])
    system_prompt = data.get('system', '')

    # Check if this is the user chatting with Aasha, or an invisible background task
    is_chat = "You are Aasha" in system_prompt

    try:
        # Save chat messages to DB
        if is_chat and messages:
            latest_user_msg = messages[-1]['content']
            new_user_db_msg = ChatMessage(role='user', content=latest_user_msg)
            db.session.add(new_user_db_msg)
            db.session.commit()

        gemini_history = []
        for msg in messages:
            role = "model" if msg['role'] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [msg['content']]})

        if gemini_history and gemini_history[0]['role'] == 'model':
            gemini_history.pop(0)

        if not gemini_history:
            return jsonify({"error": "No valid user messages to process"}), 400

        # 🚨 WE ARE NOW USING THE STABLE 2.0 MODEL 🚨
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_prompt
        )

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=data.get('max_tokens', 800)
        )

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        latest_prompt = gemini_history.pop()

        # Use chat for Aasha, but use generate_content for quick scoring
        if is_chat:
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(
                latest_prompt['parts'][0],
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        else:
            response = model.generate_content(
                latest_prompt['parts'][0],
                generation_config=generation_config,
                safety_settings=safety_settings
            )

        reply_text = response.text

        if is_chat:
            new_bot_db_msg = ChatMessage(role='model', content=reply_text)
            db.session.add(new_bot_db_msg)
            db.session.commit()

        return jsonify({"reply": reply_text})

    except Exception as e:
        print(f"Server Error: {e}")
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
        return jsonify({"error": "Failed to save assessment"}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    # Change this line
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)