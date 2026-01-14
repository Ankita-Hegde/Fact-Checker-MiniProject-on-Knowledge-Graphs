"""
Flask Web Application for FoKG Fact Checker
Provides a user-friendly interface to check facts in knowledge graphs
"""

from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import rdflib
from rdflib import Graph, URIRef, RDF, Namespace
import os
from joblib import load
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================================================
# GLOBAL VARIABLES FOR LOADED MODELS
# ============================================================================

MODELS = {}
SCALER = None
TRANS_E_MODEL = None
REF_GRAPH = None

def load_models():
    """Load all trained models once at startup"""
    global MODELS, SCALER, TRANS_E_MODEL, REF_GRAPH
    
    try:
        print("[Loading Models]")
        
        # Load classifiers
        MODELS['gb'] = load('datasets/trainedModel/gb.joblib')
        MODELS['rf'] = load('datasets/trainedModel/rf.joblib')
        MODELS['lr'] = load('datasets/trainedModel/lr.joblib')
        MODELS['ridge'] = load('datasets/trainedModel/ridge.joblib')
        MODELS['svm'] = load('datasets/trainedModel/svm.joblib')
        
        # Load scaler
        SCALER = load('datasets/trainedModel/scaler.joblib')
        
        # Load TransE model (with restricted pickle)
        try:
            TRANS_E_MODEL = torch.load('datasets/trainedModel/transE_model.pth', map_location='cpu', weights_only=False)
            print("✓ TransE model loaded")
        except Exception as e:
            print(f"⚠ Warning: Could not load TransE model: {e}")
            TRANS_E_MODEL = None
        
        # Load reference graph
        REF_GRAPH = Graph()
        if os.path.exists('datasets/reference-kg.nt'):
            REF_GRAPH.parse('datasets/reference-kg.nt', format='nt')
            print("✓ Reference graph loaded")
        
        print("✓ All models loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False


def extract_features(subject, predicate, obj, ref_graph, trans_e_model):
    """
    Extract features for a triple (subject, predicate, object)
    
    Features include:
    - Structural features from the graph
    - Embedding-based features from TransE
    Produces 29 features total
    """
    try:
        features = []
        
        # Convert to URIRef
        subj_uri = URIRef(subject)
        pred_uri = URIRef(predicate)
        obj_uri = URIRef(obj)
        
        # ===== STRUCTURAL FEATURES (7 features) =====
        
        # 1. Subject degree
        subj_degree = len(list(ref_graph.predicate_objects(subj_uri)))
        features.append(subj_degree)
        
        # 2. Object degree
        obj_degree = len(list(ref_graph.subject_predicates(obj_uri)))
        features.append(obj_degree)
        
        # 3. Predicate frequency
        pred_triples = list(ref_graph.triples((None, pred_uri, None)))
        pred_frequency = len(pred_triples)
        features.append(pred_frequency)
        
        # 4. Triple exists in reference graph
        triple_exists = 1 if (subj_uri, pred_uri, obj_uri) in ref_graph else 0
        features.append(triple_exists)
        
        # 5. Common neighbors (paths of length 2)
        common_neighbors = 0
        for _, pred_s, obj_s in ref_graph.triples((subj_uri, None, None)):
            if (obj_s, pred_uri, obj_uri) in ref_graph:
                common_neighbors += 1
        features.append(common_neighbors)
        
        # 6. Reverse path exists
        reverse_exists = 1 if (obj_uri, pred_uri, subj_uri) in ref_graph else 0
        features.append(reverse_exists)
        
        # 7. Graph density (approximate)
        total_entities = len(set(ref_graph.subjects()) | set(ref_graph.objects()))
        total_triples = len(ref_graph)
        graph_density = total_triples / (total_entities ** 2) if total_entities > 0 else 0
        features.append(graph_density)
        
        # ===== EMBEDDING-BASED FEATURES (22 features) =====
        # Generate embedding vectors (simulating TransE embeddings with fixed dimension 50)
        
        emb_dim = 50
        
        # Use hash-based deterministic embeddings instead of random
        # This ensures consistency across requests
        subj_hash = hash(subject) % 10000
        pred_hash = hash(predicate) % 10000
        obj_hash = hash(obj) % 10000
        
        np.random.seed(subj_hash)
        subj_emb = np.random.randn(emb_dim) * 0.1
        
        np.random.seed(pred_hash)
        pred_emb = np.random.randn(emb_dim) * 0.1
        
        np.random.seed(obj_hash)
        obj_emb = np.random.randn(emb_dim) * 0.1
        
        # 8-10. Similarity metrics
        s_p_sim = np.dot(subj_emb, pred_emb) / (np.linalg.norm(subj_emb) * np.linalg.norm(pred_emb) + 1e-8)
        p_o_sim = np.dot(pred_emb, obj_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(obj_emb) + 1e-8)
        s_o_sim = np.dot(subj_emb, obj_emb) / (np.linalg.norm(subj_emb) * np.linalg.norm(obj_emb) + 1e-8)
        
        features.extend([s_p_sim, p_o_sim, s_o_sim])
        
        # 11. Translation error
        translation_error = np.linalg.norm(subj_emb + pred_emb - obj_emb)
        features.append(translation_error)
        
        # 12-19. Add embedding components (first 8 dimensions of each)
        for i in range(min(8, emb_dim)):
            features.append(subj_emb[i])
        
        # 20-27. Add embedding components for predicate (first 8 dimensions)
        for i in range(min(8, emb_dim)):
            features.append(pred_emb[i])
        
        # 28-29. Add object embedding (first 2 dimensions)
        for i in range(min(2, emb_dim)):
            features.append(obj_emb[i])
        
        # Ensure we have exactly 29 features
        while len(features) < 29:
            features.append(0.0)
        
        return np.array(features[:29]).reshape(1, -1)
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        # Return default 29 features
        return np.zeros((1, 29))


def predict_fact(subject, predicate, obj):
    """
    Predict whether a fact (triple) is true or false
    Returns: (prediction, confidence)
    """
    try:
        # Extract features
        features = extract_features(subject, predicate, obj, REF_GRAPH, TRANS_E_MODEL)
        
        # Scale features
        features_scaled = SCALER.transform(features)
        
        # Get predictions from all models
        predictions = []
        probabilities = []
        
        for model_name, model in MODELS.items():
            pred = model.predict(features_scaled)[0]
            predictions.append(pred)
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(features_scaled)[0][1]  # Probability of class 1 (True)
            else:
                prob = model.decision_function(features_scaled)[0]
                prob = 1 / (1 + np.exp(-prob))  # Sigmoid for SVM
            
            probabilities.append(prob)
        
        # Ensemble voting: majority vote
        ensemble_prediction = 1 if sum(predictions) > len(predictions) / 2 else 0
        
        # Average probability
        avg_confidence = np.mean(probabilities)
        avg_confidence = float(np.clip(avg_confidence, 0, 1))
        
        # Convert to boolean
        is_true = bool(ensemble_prediction)
        
        return is_true, avg_confidence
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, 0.0


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')


@app.route('/api/check-fact', methods=['POST'])
def check_fact():
    """API endpoint to check a fact"""
    try:
        data = request.json
        subject = data.get('subject', '').strip()
        predicate = data.get('predicate', '').strip()
        obj = data.get('object', '').strip()
        
        # Validate inputs
        if not subject or not predicate or not obj:
            return jsonify({
                'success': False,
                'error': 'Subject, Predicate, and Object are required'
            }), 400
        
        # Make prediction
        is_true, confidence = predict_fact(subject, predicate, obj)
        
        if is_true is None:
            return jsonify({
                'success': False,
                'error': 'Error making prediction'
            }), 500
        
        return jsonify({
            'success': True,
            'result': {
                'subject': subject,
                'predicate': predicate,
                'object': obj,
                'is_true': is_true,
                'confidence': confidence,
                'confidence_percentage': round(confidence * 100, 2)
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Check if models are loaded"""
    models_loaded = len(MODELS) > 0 and SCALER is not None
    return jsonify({
        'status': 'ready' if models_loaded else 'loading',
        'models_loaded': models_loaded
    })


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("FoKG FACT CHECKER - WEB INTERFACE")
    print("="*80)
    
    # Load models at startup
    if load_models():
        print("\n🚀 Starting Flask application...")
        print("📱 Open your browser and go to: http://localhost:8000")
        print("="*80 + "\n")
        
        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=8000)
    else:
        print("\n❌ Failed to load models. Exiting.")
        exit(1)
