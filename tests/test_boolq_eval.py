import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'boolq_eval.py'
SPEC = importlib.util.spec_from_file_location('boolq_eval', MODULE_PATH)
BOOLQ_EVAL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BOOLQ_EVAL)

compute_boolq_confusion_counts = BOOLQ_EVAL.compute_boolq_confusion_counts
compute_average_boolq_margin = BOOLQ_EVAL.compute_average_boolq_margin
normalize_boolq_answer = BOOLQ_EVAL.normalize_boolq_answer


def test_normalize_boolq_answer_accepts_common_labels():
    assert normalize_boolq_answer('Yes') is True
    assert normalize_boolq_answer('yes.') is True
    assert normalize_boolq_answer('No:') is False


def test_compute_boolq_confusion_counts_uses_yes_as_positive_class():
    data = [
        {'choices': ['No', 'Yes']},
        {'choices': ['Yes', 'No']},
        {'choices': ['No', 'Yes']},
        {'choices': ['Yes', 'No']},
    ]
    details = [
        {'index': 0, 'pred_idx': 1, 'gold_idx': 1},  # TP
        {'index': 1, 'pred_idx': 1, 'gold_idx': 1},  # TN
        {'index': 2, 'pred_idx': 1, 'gold_idx': 0},  # FP
        {'index': 3, 'pred_idx': 1, 'gold_idx': 0},  # FN
    ]

    confusion = compute_boolq_confusion_counts(details, data)

    assert confusion == {'tp': 1, 'tn': 1, 'fp': 1, 'fn': 1}


def test_compute_average_boolq_margin_uses_yes_minus_no_logp():
    data = [
        {'choices': ['No', 'Yes']},
        {'choices': ['Yes', 'No']},
    ]
    details = [
        {'index': 0, 'choice_logps': [-3.0, -1.0]},
        {'index': 1, 'choice_logps': [-0.5, -2.0]},
    ]

    average_margin = compute_average_boolq_margin(details, data)

    assert average_margin == 1.75