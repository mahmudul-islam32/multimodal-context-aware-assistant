import pytest

from mcaa.config import load_scenario
from mcaa.rag import DocumentStore, RAGPipeline


@pytest.fixture(scope="session")
def tech_pipeline():
    sc = load_scenario("technician")
    st = DocumentStore()
    st.ingest_dir(sc.docs_path())
    return RAGPipeline(st, sc.escalation)


@pytest.fixture(scope="session")
def small_har():
    from mcaa.har.models import FeatureModel
    from mcaa.sensors.synthetic import make_dataset

    X, y, g = make_dataset(n_subjects=4, seconds_per_activity=15, seed=1)
    return FeatureModel(n_estimators=60, seed=0).fit(X, y), (X, y, g)
