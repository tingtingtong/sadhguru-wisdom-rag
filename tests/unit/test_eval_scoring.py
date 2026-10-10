from evaluation.runner import _compute_overall_score


def test_page_grounded_metrics_drive_retrieval_score():
    report = {"retrieval": {
        "aggregate": {"avg_context_precision": 1, "avg_context_recall": 1, "avg_mrr": 1, "avg_ndcg": 1},
        "page_grounded": {"hit@5": 0.8, "hit@1": 0.6, "mrr": 0.7},
    }}
    got = _compute_overall_score(report)["category_scores"]["retrieval"]
    assert abs(got - (0.8 * 0.5 + 0.7 * 0.3 + 0.6 * 0.2) * 100) < 0.1  # keyword metrics ignored


def test_falls_back_to_keyword_metrics_without_page_eval():
    report = {"retrieval": {"aggregate": {"avg_context_precision": 0.5, "avg_context_recall": 0.5, "avg_mrr": 0.5, "avg_ndcg": 0.5}}}
    assert abs(_compute_overall_score(report)["category_scores"]["retrieval"] - 50) < 0.1
