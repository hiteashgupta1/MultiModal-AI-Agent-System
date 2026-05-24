from rouge_score import rouge_scorer
from bert_score import score


def compute_rouge(reference, summary):

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    scores = scorer.score(reference, summary)

    return {
        "rouge1": round(scores["rouge1"].fmeasure, 3),
        "rouge2": round(scores["rouge2"].fmeasure, 3),
        "rougeL": round(scores["rougeL"].fmeasure, 3)
    }


def compute_bertscore(reference, summary):

    P, R, F1 = score(
        [summary],
        [reference],
        lang="en",
        verbose=False
    )

    return {
        "bertscore_precision": round(P.mean().item(), 3),
        "bertscore_recall": round(R.mean().item(), 3),
        "bertscore_f1": round(F1.mean().item(), 3)
    }