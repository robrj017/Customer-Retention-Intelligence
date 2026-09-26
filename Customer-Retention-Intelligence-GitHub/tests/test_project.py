from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
def test_project_structure():
    required_paths = [
        BASE_DIR / "app" / "app.py",
        BASE_DIR / "knowledge_base",
        BASE_DIR / "rag" / "chunks.csv",
        BASE_DIR / "rag" / "chunk_embeddings.npy",
        BASE_DIR / "reports" / "customer_risk_scores.csv",
        BASE_DIR / "README.md",
        BASE_DIR / "requirements.txt"
    ]
    for path in required_paths:
        assert path.exists(), f"Missing project file: {path}"
def test_rag_embeddings():
    import numpy as np
    embeddings = np.load(
        BASE_DIR / "rag" / "chunk_embeddings.npy"
    )
    assert embeddings.ndim == 2
    assert embeddings.shape[1] == 384
def test_customer_risk_data():
    import pandas as pd
    data = pd.read_csv(
        BASE_DIR / "reports" / "customer_risk_scores.csv"
    )
    assert len(data) == 7043
    assert "customerID" in data.columns
    assert "RiskScore" in data.columns
    assert "RiskLevel" in data.columns