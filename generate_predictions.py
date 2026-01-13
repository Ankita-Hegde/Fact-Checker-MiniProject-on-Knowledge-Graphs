"""
Generate result_1.ttl using saved v6 models
No retraining needed - just loads models and makes predictions
"""

import torch
import numpy as np
import rdflib
from rdflib import Graph, URIRef, RDF
import os
from joblib import load

print("="*80)
print("GENERATING RESULTS FROM SAVED V6 MODELS")
print("="*80)

# ============================================================================
# STEP 1: Load Saved Models
# ============================================================================

print("\n[1] Loading saved models...")

try:
    gb_model = load('datasets/trainedModel/gb.joblib')
    print("✓ Gradient Boosting")
    
    rf_model = load('datasets/trainedModel/rf.joblib')
    print("✓ Random Forest")
    
    lr_model = load('datasets/trainedModel/lr.joblib')
    print("✓ Logistic Regression")
    
    ridge_model = load('datasets/trainedModel/ridge.joblib')
    print("✓ Ridge")
    
    svm_model = load('datasets/trainedModel/svm.joblib')
    print("✓ SVM")
    
    scaler = load('datasets/trainedModel/scaler.joblib')
    print("✓ Scaler")
except Exception as e:
    print(f"ERROR loading models: {e}")
    exit(1)

# ============================================================================
# STEP 2: Load TransE Model
# ============================================================================

print("\n[2] Loading TransE model...")

try:
    checkpoint = torch.load('datasets/trainedModel/transE_model.pth', map_location='cpu')
    
    from pykeen.models import TransE
    from pykeen.losses import MarginRankingLoss
    
    triples_factory = checkpoint['triples_factory']
    embedding_dim = checkpoint['embedding_dim']
    
    # Reconstruct model
    model = TransE(
        triples_factory=triples_factory,
        embedding_dim=embedding_dim,
        scoring_fct_norm=1,
        loss=MarginRankingLoss(margin=1.5),
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ TransE model (dim={embedding_dim})")
except Exception as e:
    print(f"ERROR loading TransE: {e}")
    exit(1)

# ============================================================================
# STEP 3: Load Reference KG for Features
# ============================================================================

print("\n[3] Loading reference knowledge graph...")

try:
    ref_graph = Graph()
    ref_graph.parse('datasets/reference-kg.nt', format='nt')
    
    all_triples = set()
    for s, p, o in ref_graph:
        if isinstance(s, rdflib.URIRef) and isinstance(p, rdflib.URIRef) and isinstance(o, rdflib.URIRef):
            all_triples.add((str(s), str(p), str(o)))
    
    print(f"✓ Loaded {len(all_triples)} reference triples")
except Exception as e:
    print(f"WARNING: Could not load reference KG: {e}")
    all_triples = set()

# ============================================================================
# STEP 4: Load Test Data
# ============================================================================

print("\n[4] Loading test data...")

TRUTH_VALUE_PRED = URIRef("http://swc2017.aksw.org/hasTruthValue")

def load_reified_facts(nt_file, load_truth=True):
    g = rdflib.Graph()
    g.parse(nt_file, format="nt")
    facts = []
    for fact_iri in g.subjects(predicate=RDF.type, object=RDF.Statement):
        subj = g.value(subject=fact_iri, predicate=RDF.subject)
        pred = g.value(subject=fact_iri, predicate=RDF.predicate)
        obj = g.value(subject=fact_iri, predicate=RDF.object)
        if subj is None or pred is None or obj is None:
            continue
        if load_truth:
            truth_val = g.value(subject=fact_iri, predicate=TRUTH_VALUE_PRED)
            if truth_val is None:
                continue
            truth_val = float(truth_val.toPython())
            facts.append((str(fact_iri), str(subj), str(pred), str(obj), truth_val))
        else:
            facts.append((str(fact_iri), str(subj), str(pred), str(obj), None))
    return facts

test_facts = load_reified_facts('datasets/test.nt', load_truth=False)
print(f"✓ Loaded {len(test_facts)} test facts")

# ============================================================================
# STEP 5: Extract Features
# ============================================================================

print("\n[5] Extracting features for test data...")

entity_to_id = triples_factory.entity_to_id
relation_to_id = triples_factory.relation_to_id

def get_embeddings(subj, pred, obj):
    """Get embeddings for a triple"""
    if subj not in entity_to_id or obj not in entity_to_id or pred not in relation_to_id:
        return np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim)
    
    s_id = entity_to_id[subj]
    p_id = relation_to_id[pred]
    o_id = entity_to_id[obj]
    
    with torch.no_grad():
        s_emb = model.entity_representations[0](indices=torch.tensor([s_id]))
        p_emb = model.relation_representations[0](indices=torch.tensor([p_id]))
        o_emb = model.entity_representations[0](indices=torch.tensor([o_id]))
    
    return (s_emb[0].cpu().numpy(),
            p_emb[0].cpu().numpy(),
            o_emb[0].cpu().numpy())

def extract_features(s, p, o):
    """Extract comprehensive features"""
    s_emb, p_emb, o_emb = get_embeddings(s, p, o)
    eps = 1e-8
    
    s_norm = np.linalg.norm(s_emb) + eps
    p_norm = np.linalg.norm(p_emb) + eps
    o_norm = np.linalg.norm(o_emb) + eps
    
    s_norm_vec = s_emb / s_norm
    p_norm_vec = p_emb / p_norm
    o_norm_vec = o_emb / o_norm
    
    feat = []
    
    # TransE scores
    feat.append(np.linalg.norm(s_emb + p_emb - o_emb, ord=1))
    feat.append(np.linalg.norm(s_emb + p_emb - o_emb, ord=2))
    feat.append(np.linalg.norm(s_emb + p_emb - o_emb, ord=np.inf))
    
    # Distance metrics
    feat.append(np.linalg.norm(s_emb - o_emb, ord=1))
    feat.append(np.linalg.norm(s_emb - o_emb, ord=2))
    
    # Similarity scores
    feat.append(np.dot(s_norm_vec, o_norm_vec))
    feat.append(np.dot(s_norm_vec, p_norm_vec))
    feat.append(np.dot(p_norm_vec, o_norm_vec))
    
    # Magnitude features
    feat.append(np.log(s_norm))
    feat.append(np.log(p_norm))
    feat.append(np.log(o_norm))
    feat.append(s_norm)
    feat.append(p_norm)
    feat.append(o_norm)
    
    # Mean/Max absolute values
    feat.append(np.mean(np.abs(s_emb)))
    feat.append(np.mean(np.abs(p_emb)))
    feat.append(np.mean(np.abs(o_emb)))
    feat.append(np.max(np.abs(s_emb)))
    feat.append(np.max(np.abs(p_emb)))
    feat.append(np.max(np.abs(o_emb)))
    
    # Dot products
    feat.append(np.dot(s_emb, p_emb))
    feat.append(np.dot(p_emb, o_emb))
    feat.append(np.dot(s_emb, o_emb))
    
    # Tensor contractions
    feat.append(np.sum(s_emb * p_emb * o_emb))
    feat.append(np.sum((s_emb + p_emb) * o_emb))
    
    # Structural features
    in_kb = 1.0 if (s, p, o) in all_triples else 0.0
    feat.append(in_kb)
    
    reverse_exists = 1.0 if (o, p, s) in all_triples else 0.0
    feat.append(reverse_exists)
    
    # Entity connectivity
    s_count = sum(1 for (s2, _, _) in all_triples if s2 == s) if all_triples else 0
    o_count = sum(1 for (_, _, o2) in all_triples if o2 == o) if all_triples else 0
    feat.append(np.log(s_count + 1))
    feat.append(np.log(o_count + 1))
    
    return np.array(feat, dtype=np.float32)

# Extract features
print("Computing features...")
X_test = np.array([extract_features(s, p, o) for _, s, p, o, _ in test_facts])
test_fact_iris = [fact_iri for fact_iri, _, _, _, _ in test_facts]

print(f"✓ Feature matrix: {X_test.shape}")

# ============================================================================
# STEP 6: Scale Features
# ============================================================================

print("\n[6] Scaling features...")

X_test_scaled = scaler.transform(X_test)
print(f"✓ Scaled")

# ============================================================================
# STEP 7: Make Predictions
# ============================================================================

print("\n[7] Making predictions with ensemble...")

gb_probs = gb_model.predict_proba(X_test_scaled)[:, 1]
print(f"✓ GB: [{gb_probs.min():.4f}, {gb_probs.max():.4f}]")

rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
print(f"✓ RF: [{rf_probs.min():.4f}, {rf_probs.max():.4f}]")

lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
print(f"✓ LR: [{lr_probs.min():.4f}, {lr_probs.max():.4f}]")

ridge_probs = (ridge_model.decision_function(X_test_scaled) + 1) / 2
ridge_probs = np.clip(ridge_probs, 0, 1)
print(f"✓ Ridge: [{ridge_probs.min():.4f}, {ridge_probs.max():.4f}]")

svm_probs = svm_model.predict_proba(X_test_scaled)[:, 1]
print(f"✓ SVM: [{svm_probs.min():.4f}, {svm_probs.max():.4f}]")

# Ensemble with equal weights
test_probs = (gb_probs + rf_probs + lr_probs + ridge_probs + svm_probs) / 5.0
print(f"\n✓ Ensemble: [{test_probs.min():.4f}, {test_probs.max():.4f}]")

# ============================================================================
# STEP 8: Write Results
# ============================================================================

print("\n[8] Writing results to result_1.ttl...")

os.makedirs('resultFile', exist_ok=True)
with open('resultFile/predictedResult.ttl', 'w', encoding='utf-8') as f:
    for fact_iri, score in zip(test_fact_iris, test_probs):
        line = f'<{fact_iri}> <http://swc2017.aksw.org/hasTruthValue> "{float(score)}"^^<http://www.w3.org/2001/XMLSchema#double> .\n'
        f.write(line)

print(f"✓ Written {len(test_probs)} predictions to resultFile/predictedResult.ttl")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"\nResult file: resultFile/predictedResult.ttl")
print(f"Predictions: {len(test_probs)}")
print(f"Score range: [{test_probs.min():.4f}, {test_probs.max():.4f}]")
print(f"Mean score: {test_probs.mean():.4f}")
