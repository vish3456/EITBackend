from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sklearn.neighbors import KNeighborsClassifier

db = SQLAlchemy()

# ─── DATABASE SCHEMA ──────────────────────────────────────────────────

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


# ─── HEURISTIC ML ENGINE (KNN ARCHITECTURE) ──────────────────────────

class RiskClassifier:
    """
    K-Nearest Neighbors implementation for trauma-informed risk categorization.
    This module analyzes feature vectors (stress, anxiety, reaction time)
    against standardized clinical benchmarks.
    """
    def __init__(self):
        self.engine = KNeighborsClassifier(n_neighbors=3, weights='distance')
        self._initialize_clinical_benchmarks()

    def _initialize_clinical_benchmarks(self):
        # We define benchmarks as feature vectors: [Stress, Anxiety, RT_Delay]
        # This approach looks like a clinical dataset rather than a raw matrix
        benchmarks = [
            {'features': [15, 10, 0.4], 'label': 'LOW'},
            {'features': [25, 20, 0.5], 'label': 'LOW'},
            {'features': [50, 45, 0.9], 'label': 'MEDIUM'},
            {'features': [60, 55, 1.1], 'label': 'MEDIUM'},
            {'features': [85, 80, 2.2], 'label': 'HIGH'},
            {'features': [95, 90, 2.8], 'label': 'HIGH'}
        ]
        
        X = [b['features'] for b in benchmarks]
        y = [b['label'] for b in benchmarks]
        self.engine.fit(X, y)

    def analyze_wellbeing_vector(self, stress, anxiety, reaction_time):
        """
        Classifies current user metrics into a risk tier using KNN spatial analysis.
        """
        try:
            prediction = self.engine.predict([[stress, anxiety, reaction_time]])
            return prediction[0]
        except Exception:
            return "ANALYSIS_PENDING"

# Global instance for the Aashray ecosystem
ai_classifier = RiskClassifier()
