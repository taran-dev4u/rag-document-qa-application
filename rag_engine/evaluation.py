from typing import Dict, List, Set, Tuple

from .chunking import DocumentChunk


def evaluate_retrieval(
    retrieved_results: List[List[Tuple[DocumentChunk, float]]],
    ground_truth_doc_ids: List[Set[str]],
    k: int = 5,
) -> Dict[str, float]:
    """Computes standard Information Retrieval metrics across evaluation query runs.

    Metrics computed:
    - Hit Rate @ K: Fraction of queries with at least one relevant document retrieved.
    - MRR (Mean Reciprocal Rank): Mean of reciprocal rank of the first relevant document.
    - Precision @ K: Mean proportion of retrieved chunks belonging to relevant documents.
    - Recall @ K: Mean fraction of relevant documents retrieved.
    """
    if not retrieved_results or not ground_truth_doc_ids:
        return {"hit_rate_at_k": 0.0, "mrr": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0}

    n_queries = len(retrieved_results)
    hits = 0
    reciprocal_ranks = []
    precisions = []
    recalls = []

    for results, truth_set in zip(retrieved_results, ground_truth_doc_ids):
        top_k_chunks = results[:k]
        top_k_doc_ids = [chunk.doc_id for chunk, _ in top_k_chunks]

        # Hit rate
        hit = any(doc_id in truth_set for doc_id in top_k_doc_ids)
        if hit:
            hits += 1

        # MRR
        first_rank = 0
        for rank, (chunk, _) in enumerate(top_k_chunks, start=1):
            if chunk.doc_id in truth_set:
                first_rank = rank
                break
        reciprocal_ranks.append(1.0 / first_rank if first_rank > 0 else 0.0)

        # Precision & Recall
        if top_k_doc_ids:
            relevant_retrieved = sum(1 for d in top_k_doc_ids if d in truth_set)
            precisions.append(relevant_retrieved / len(top_k_doc_ids))
        else:
            precisions.append(0.0)

        unique_truth_retrieved = len(set(top_k_doc_ids).intersection(truth_set))
        recalls.append(unique_truth_retrieved / max(1, len(truth_set)))

    return {
        "queries_evaluated": n_queries,
        "k": k,
        "hit_rate_at_k": round(hits / n_queries, 4),
        "mrr": round(sum(reciprocal_ranks) / n_queries, 4),
        "precision_at_k": round(sum(precisions) / n_queries, 4),
        "recall_at_k": round(sum(recalls) / n_queries, 4),
    }
