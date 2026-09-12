from rag_engine.chunking import DocumentChunk
from rag_engine.evaluation import evaluate_retrieval


def test_evaluate_retrieval_perfect_hit():
    c1 = DocumentChunk("c1", "doc_true", "Text here", 0, 9)
    c2 = DocumentChunk("c2", "doc_false", "Other text", 0, 10)

    retrieved = [[(c1, 0.9), (c2, 0.4)]]
    ground_truth = [{"doc_true"}]

    metrics = evaluate_retrieval(retrieved, ground_truth, k=2)
    assert metrics["hit_rate_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["precision_at_k"] == 0.5
    assert metrics["recall_at_k"] == 1.0


def test_evaluate_retrieval_miss():
    c1 = DocumentChunk("c1", "doc_wrong", "Irrelevant", 0, 10)
    retrieved = [[(c1, 0.5)]]
    ground_truth = [{"doc_target"}]

    metrics = evaluate_retrieval(retrieved, ground_truth, k=1)
    assert metrics["hit_rate_at_k"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["precision_at_k"] == 0.0
    assert metrics["recall_at_k"] == 0.0


def test_evaluate_empty_inputs():
    metrics = evaluate_retrieval([], [])
    assert metrics["hit_rate_at_k"] == 0.0
    assert metrics["mrr"] == 0.0
