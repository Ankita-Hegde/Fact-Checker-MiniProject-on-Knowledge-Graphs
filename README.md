# FoKG Fact Checker

A machine learning-based system for fact checking in knowledge graphs using advanced embedding models and ensemble learning techniques.

## 📋 Problem Statement

Knowledge graphs contain vast amounts of factual information, but verifying the accuracy of triples (facts) is challenging, especially when dealing with incomplete or noisy data. Manual fact verification is labor-intensive and not scalable. 

**Problem:** Given a knowledge graph with reference facts, how can we automatically verify the correctness of new or uncertain triples?

## 💡 Solution Approach

The FoKG Fact Checker employs an advanced ensemble machine learning approach combining multiple techniques:

### Core Components:

1. **Embedding Models**
   - **TransE**: Translational embedding model that learns low-dimensional representations of entities and relations in the knowledge graph
   - Captures semantic relationships between entities based on graph structure

2. **Feature Engineering**
   - **Structural Features**: Extracted from the knowledge graph structure (degree, paths, connectivity)
   - **Semantic Features**: Derived from TransE embeddings and entity-relation patterns
   - **Comprehensive Feature Set**: Multi-faceted representation of each triple

3. **Ensemble Learning Stack**
   - Multiple base learners trained on engineered features:
     - Gradient Boosting Classifier
     - Random Forest Classifier
     - Logistic Regression
     - Ridge Classifier
     - Support Vector Machine (SVM)
   - Meta-learner for combining predictions from base models
   - Optimal threshold selection for binary classification

4. **Optimization & Tuning**
   - Hyperparameter tuning using Optuna
   - Cross-validation with StratifiedKFold
   - Probability calibration for robust predictions

## 🔍 Solution Explanation

The system works in two phases:

### Phase 1: Training (factCheck.py)
1. Load reference knowledge graph and training data
2. Train TransE embedding model on reference facts
3. Extract structural and semantic features for each triple
4. Train multiple base classifiers (GB, RF, LR, Ridge, SVM) with hyperparameter optimization
5. Create ensemble stack with meta-learner
6. Calibrate probability predictions for better threshold selection
7. Save all trained models to `datasets/trainedModel/`

### Phase 2: Prediction (generate_predictions.py)
1. Load pre-trained embedding model (TransE) and ensemble models
2. Extract features for test triples using the same methodology
3. Generate predictions using the ensemble
4. Output results in TTL (RDF) format

### Why This Approach?
- **TransE Embeddings**: Captures semantic similarity without explicit features
- **Multiple Base Learners**: Different algorithms learn different aspects of the data
- **Ensemble Methods**: Combines strengths of multiple models, reducing overfitting
- **Feature Engineering**: Domain-specific features improve interpretability and performance
- **Calibration**: Ensures probability predictions are reliable for decision-making

## 📁 Project Structure

```
FoKG Fact Checker/
├── factCheck.py                    # Main training script
├── generate_predictions.py         # Prediction script using trained models
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── datasets/
│   ├── reference-kg.nt            # Reference knowledge graph
│   ├── train.nt                   # Training triples
│   ├── test.nt                    # Test triples for prediction
│   └── trainedModel/              # Pre-trained models (do NOT delete)
│       ├── transE_model.pth       # TransE embedding model
│       ├── gb.joblib              # Gradient Boosting model
│       ├── rf.joblib              # Random Forest model
│       ├── lr.joblib              # Logistic Regression model
│       ├── ridge.joblib           # Ridge Classifier model
│       ├── svm.joblib             # SVM model
│       └── scaler.joblib          # Feature scaler
└── resultFile/
    └── result.ttl                 # Output predictions in RDF format
```

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd "FoKG Fact Checker"
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip3 install -r requirements.txt
```

This will install:
- `torch` - Deep learning framework for TransE embeddings
- `pykeen` - Knowledge graph embedding library
- `scikit-learn` - Machine learning algorithms and utilities
- `rdflib` - RDF/Knowledge graph handling
- `networkx` - Graph processing
- `pandas` - Data manipulation
- `joblib` - Model serialization
- And other required packages

## 🚀 Execution Steps

### Option A: Generate Predictions Using Pre-trained Models (Recommended ⭐)

The trained models are already provided in `datasets/trainedModel/`. You can immediately generate predictions without training:

```bash
python3 generate_predictions.py
```

**Output:** Generates `resultFile/result.ttl` with fact-checking predictions for your test data.

**Time Required:** ~2-5 minutes (depends on test dataset size)

### Option B: Train New Models (Advanced)

⚠️ **Note:** Model training takes **several hours** (3-12+ hours depending on hardware and dataset size). This step is optional if you want to use the pre-trained models.

```bash
python3 factCheck.py
```

**What happens:**
1. Loads reference knowledge graph
2. Trains TransE embedding model (most time-consuming part)
3. Extracts features from training data
4. Trains and optimizes 5 base classifiers
5. Creates ensemble stack
6. Saves all models to `datasets/trainedModel/`

**Time Required:** 3-12+ hours (depending on your CPU/GPU and dataset size)

**Hardware Recommendations:**
- **GPU**: Recommended for faster TransE training 
- **CPU**: Will work but be significantly slower
- **RAM**: Minimum 8GB, 16GB+ recommended

## 📊 Expected Output

After running `generate_predictions.py`, check `resultFile/predictedResult.ttl`:

```ttl
@prefix swc2017: <http://swc2017.aksw.org/> .

<http://example.org/fact1> swc2017:hasTruthValue true ;
                           swc2017:confidence 0.95 .

<http://example.org/fact2> swc2017:hasTruthValue false ;
                           swc2017:confidence 0.87 .
```

Each fact gets:
- A truth value (true/false)
- A confidence score (0.0 to 1.0)

## 🔑 Key Features

✅ **Pre-trained Models**: No need to train from scratch  
✅ **Fast Inference**: Generate predictions in minutes, not hours  
✅ **Ensemble Method**: Combines 5 different ML algorithms  
✅ **Knowledge Graph Native**: Works with RDF/N-Triples format  
✅ **Scalable**: Can handle large knowledge graphs  
✅ **Interpretable**: Feature-based approach provides transparency  

## 📝 Input Data Format

Your input files should be in N-Triples format (`.nt`):

```n-triples
<http://example.org/subject> <http://example.org/predicate> <http://example.org/object> .
<http://example.org/person1> <http://example.org/knows> <http://example.org/person2> .
```

## 🆘 Troubleshooting

**Issue:** "Module not found" error
- **Solution:** Ensure you've installed all dependencies with `pip3 install -r requirements.txt`

**Issue:** TransE model not found
- **Solution:** Make sure `datasets/trainedModel/transE_model.pth` exists. If not, run training with `factCheck.py`

**Issue:** Out of memory errors
- **Solution:** Reduce dataset size or increase available RAM. Consider using a GPU for faster processing.

**Issue:** Slow execution
- **Solution:** Use a GPU-enabled machine. Training/inference on CPU is significantly slower.

## 📚 Dependencies

All required packages are listed in `requirements.txt`:
- **numpy** - Numerical computations
- **torch** - Deep learning framework
- **pykeen** - Knowledge graph embedding
- **scikit-learn** - Machine learning algorithms
- **rdflib** - RDF handling
- **networkx** - Graph algorithms
- **pandas** - Data manipulation
- **joblib** - Model serialization

## 🎯 Next Steps

1. Ensure your test data is in `datasets/test.nt`
2. Run `python3 generate_predictions.py`
3. Check results in `resultFile/result.ttl`
4. Use the truth values and confidence scores for downstream applications

## ⏱️ Important: Training Time Notice

⚠️ **If you decide to retrain models** using `factCheck.py`:
- **Total training time:** 3-12+ hours (depending on hardware)
- **TransE training:** This is the most time-consuming step (1-8+ hours)
- **Ensemble training:** Several hours for optimization and cross-validation
- **Pre-trained models are provided** to avoid this wait time!

**Recommendation:** Use the provided pre-trained models in `datasets/trainedModel/` for immediate fact-checking. Only retrain if you have new training data or want to optimize further.


