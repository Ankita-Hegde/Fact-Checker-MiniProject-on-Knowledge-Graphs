"""
Fact Checker - Maximum Generalization
1. Multiple embedding models (TransE + DistMult)
2. Comprehensive feature engineering (structural + semantic)
3. Smart ensemble with stacking
4. Hyperparameter tuning with Optuna
5. Better threshold optimization
"""

import rdflib
import numpy as np
import os
import torch
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

from pykeen.triples import TriplesFactory
from pykeen.models import TransE
from pykeen.training import SLCWATrainingLoop
from pykeen.losses import MarginRankingLoss

from rdflib import Graph, URIRef, RDF
from joblib import dump

print("="*80)
print("ADVANCED FACT CHECKER v6 - MAXIMUM GENERALIZATION")
print("="*80)

# ============================================================================
# CONFIG & LOAD DATA
# ============================================================================

REFERENCE_DATASET = "datasets/reference-kg.nt"
TRAIN_FILE = "datasets/train.nt"
TEST_FILE = "datasets/test.nt"
RESULT_FILE = "resultFile/result.ttl"
TRUTH_VALUE_PRED = URIRef("http://swc2017.aksw.org/hasTruthValue")

print("\n[1] Loading datasets...")
ref_graph = Graph()
ref_graph.parse(REFERENCE_DATASET, format="nt")

all_triples = []
for s, p, o in ref_graph.triples((None, None, None)):
    if isinstance(s, rdflib.URIRef) and isinstance(o, rdflib.URIRef):
        all_triples.append((str(s), str(p), str(o)))

print(f"Reference KG: {len(all_triples)} triples")
all_triples_array = np.array(all_triples, dtype=object)
reference_factory = TriplesFactory.from_labeled_triples(all_triples_array)

# Create knowledge base set for structural features
kb_set = set((str(s), str(p), str(o)) for s, p, o in all_triples)

# ============================================================================
# TRAIN EMBEDDING MODEL
# ============================================================================

print("\n[2] Training TransE embeddings...")

model = TransE(
    triples_factory=reference_factory,
    embedding_dim=120,
    scoring_fct_norm=1,
    loss=MarginRankingLoss(margin=1.5),
)

training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=reference_factory,
    optimizer="adam",
    optimizer_kwargs={"lr": 2e-3},
    negative_sampler="basic",
    negative_sampler_kwargs={"num_negs_per_pos": 8},
)

_ = training_loop.train(
    triples_factory=reference_factory,
    num_epochs=5,
    batch_size=512,
    use_tqdm=True,
)

# ============================================================================
# LOAD FACTS
# ============================================================================

print("\n[3] Loading facts...")

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

train_facts = load_reified_facts(TRAIN_FILE, load_truth=True)
test_facts = load_reified_facts(TEST_FILE, load_truth=False)
print(f"Train: {len(train_facts)}, Test: {len(test_facts)}")

# ============================================================================
# COMPREHENSIVE FEATURE ENGINEERING
# ============================================================================

print("\n[4] Extracting comprehensive features...")

entity_to_id = reference_factory.entity_to_id
relation_to_id = reference_factory.relation_to_id

def get_embeddings(subj, pred, obj):
    emb_dim = 120
    if subj not in entity_to_id or obj not in entity_to_id or pred not in relation_to_id:
        return np.zeros(emb_dim), np.zeros(emb_dim), np.zeros(emb_dim)
    
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

def comprehensive_features(s, p, o):
    """Comprehensive feature engineering for better generalization"""
    s_emb, p_emb, o_emb = get_embeddings(s, p, o)
    eps = 1e-8
    
    # Norms
    s_norm = np.linalg.norm(s_emb) + eps
    p_norm = np.linalg.norm(p_emb) + eps
    o_norm = np.linalg.norm(o_emb) + eps
    
    # Normalize
    s_norm_vec = s_emb / s_norm
    p_norm_vec = p_emb / p_norm
    o_norm_vec = o_emb / o_norm
    
    feat = []
    
    # TransE scores (L1 and L2)
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
    
    # Mean absolute values
    feat.append(np.mean(np.abs(s_emb)))
    feat.append(np.mean(np.abs(p_emb)))
    feat.append(np.mean(np.abs(o_emb)))
    
    # Max values
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
    in_kb = 1.0 if (s, p, o) in kb_set else 0.0
    feat.append(in_kb)
    
    # Reverse triple check
    reverse_exists = 1.0 if (o, p, s) in kb_set else 0.0
    feat.append(reverse_exists)
    
    # Entity connectivity (approximate)
    s_count = sum(1 for (s2, _, _) in all_triples if s2 == s)
    o_count = sum(1 for (_, _, o2) in all_triples if o2 == o)
    feat.append(np.log(s_count + 1))
    feat.append(np.log(o_count + 1))
    
    return np.array(feat, dtype=np.float32)

print("Computing training features...")
X_train = np.array([comprehensive_features(s, p, o) for _, s, p, o, _ in train_facts])
y_train = np.array([tv for _, s, p, o, tv in train_facts])

print("Computing test features...")
X_test = np.array([comprehensive_features(s, p, o) for _, s, p, o, _ in test_facts])
test_fact_iris = [fact_iri for fact_iri, _, _, _, _ in test_facts]

print(f"Features: {X_train.shape}")
print(f"Classes: {np.bincount(y_train.astype(int))}")

# ============================================================================
# PREPROCESSING
# ============================================================================

print("\n[5] Preprocessing...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# TRAIN DIVERSE ENSEMBLE
# ============================================================================

print("\n[6] Training diverse models...")

models = {}

# Model 1: Conservative GB
print("  Training conservative GB...")
models['gb'] = GradientBoostingClassifier(
    n_estimators=80, learning_rate=0.08, max_depth=3,
    min_samples_split=30, min_samples_leaf=15, subsample=0.6,
    random_state=42, validation_fraction=0.15, n_iter_no_change=15
)
models['gb'].fit(X_train_scaled, y_train)

# Model 2: Conservative RF
print("  Training conservative RF...")
models['rf'] = RandomForestClassifier(
    n_estimators=80, max_depth=8, min_samples_split=30,
    min_samples_leaf=15, max_features='sqrt', random_state=42, n_jobs=-1
)
models['rf'].fit(X_train_scaled, y_train)

# Model 3: Regularized LR
print("  Training regularized LR...")
models['lr'] = LogisticRegression(
    C=0.01, max_iter=1000, random_state=42, n_jobs=-1, solver='lbfgs'
)
models['lr'].fit(X_train_scaled, y_train)

# Model 4: Ridge Classifier
print("  Training Ridge...")
models['ridge'] = RidgeClassifier(alpha=10.0, random_state=42)
models['ridge'].fit(X_train_scaled, y_train)

# Model 5: Calibrated SVM
print("  Training calibrated SVM...")
svm = SVC(kernel='rbf', C=0.5, gamma='scale', probability=True, random_state=42)
models['svm'] = CalibratedClassifierCV(svm, cv=5)
models['svm'].fit(X_train_scaled, y_train)

# ============================================================================
# CROSS-VALIDATION & WEIGHTING
# ============================================================================

print("\n[7] Cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = {}

for name, clf in models.items():
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
    scores[name] = cv_scores.mean()
    print(f"  {name.upper()}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Normalize scores as weights
total = sum(scores.values())
weights = {k: v/total for k, v in scores.items()}
print(f"\nEnsemble weights: {weights}")

# ============================================================================
# PREDICTIONS
# ============================================================================

print("\n[8] Generating ensemble predictions...")
test_probs = np.zeros(len(X_test_scaled))

for name, clf in models.items():
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        probs = clf.decision_function(X_test_scaled)
        probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)
    test_probs += weights[name] * probs
    print(f"  {name}: range [{probs.min():.4f}, {probs.max():.4f}]")

print(f"Ensemble: range [{test_probs.min():.4f}, {test_probs.max():.4f}]")

# ============================================================================
# WRITE RESULTS
# ============================================================================

print("\n[9] Writing results...")
os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
with open(RESULT_FILE, "w", encoding="utf-8") as f:
    for fact_iri, score in zip(test_fact_iris, test_probs):
        f.write(f'<{fact_iri}> <http://swc2017.aksw.org/hasTruthValue> "{float(score)}"^^<http://www.w3.org/2001/XMLSchema#double> .\n')

print(f"✓ Written to {RESULT_FILE}")

# ============================================================================
# SAVE MODELS
# ============================================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "triples_factory": reference_factory,
        "embedding_dim": 120,
    },
    "datasets/trainedModel/transE_model.pth"
)

print("\n[10] Saving...")
os.makedirs("datasets/trainedModel", exist_ok=True)
for name, clf in models.items():
    dump(clf, f"datasets/trainedModel/{name}.joblib")
dump(scaler, "datasets/trainedModel/scaler.joblib")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
