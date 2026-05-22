import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'boolq_eval.py'
SPEC = importlib.util.spec_from_file_location('boolq_eval', MODULE_PATH)
BOOLQ_EVAL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BOOLQ_EVAL)

compute_boolq_confusion_counts = BOOLQ_EVAL.compute_boolq_confusion_counts
compute_average_boolq_margin = BOOLQ_EVAL.compute_average_boolq_margin
compute_calibrated_boolq_accuracy = BOOLQ_EVAL.compute_calibrated_boolq_accuracy
compute_centered_boolq_score = BOOLQ_EVAL.compute_centered_boolq_score
compute_class_conditional_boolq_margin_means = BOOLQ_EVAL.compute_class_conditional_boolq_margin_means
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
        {'index': 0, 'gold_idx': 1, 'choice_logps': [-3.0, -1.0]},
        {'index': 1, 'gold_idx': 1, 'choice_logps': [-3.0, -1.0]},
        {'index': 2, 'gold_idx': 0, 'choice_logps': [-3.0, -1.0]},
        {'index': 3, 'gold_idx': 0, 'choice_logps': [-3.0, -1.0]},
    ]

    confusion = compute_boolq_confusion_counts(details, data)

    assert confusion == {'tp': 1, 'tn': 1, 'fp': 1, 'fn': 1}


def test_compute_boolq_confusion_counts_respects_tau_threshold():
    data = [
        {'choices': ['No', 'Yes']},
        {'choices': ['Yes', 'No']},
    ]
    details = [
        {'index': 0, 'gold_idx': 1, 'choice_logps': [-1.4, -1.0]},
        {'index': 1, 'gold_idx': 1, 'choice_logps': [-1.0, -1.3]},
    ]

    confusion = compute_boolq_confusion_counts(details, data, tau=0.5)

    assert confusion == {'tp': 0, 'tn': 1, 'fp': 0, 'fn': 1}


def test_compute_average_boolq_margin_uses_yes_minus_no_logp():
    data = [
        {'choices': ['No', 'Yes']},
        {'choices': ['Yes', 'No']},
    ]
    details = [
        {'index': 0, 'gold_idx': 1, 'choice_logps': [-3.0, -1.0]},
        {'index': 1, 'gold_idx': 1, 'choice_logps': [-0.5, -2.0]},
    ]

    average_margin = compute_average_boolq_margin(details, data)

    assert average_margin == 1.75


def test_compute_calibrated_boolq_accuracy_uses_tau_threshold():
    data = [
        {'choices': ['No', 'Yes']},
        {'choices': ['Yes', 'No']},
        {'choices': ['No', 'Yes']},
    ]
    details = [
        {'index': 0, 'gold_idx': 1, 'choice_logps': [-2.0, -1.0]},
        {'index': 1, 'gold_idx': 1, 'choice_logps': [-1.0, -1.4]},
        {'index': 2, 'gold_idx': 0, 'choice_logps': [-1.0, -1.2]},
    ]

    calibrated_accuracy = compute_calibrated_boolq_accuracy(details, data, tau=0.3)

    assert calibrated_accuracy == 2 / 3


def test_compute_centered_boolq_score_uses_boolq_baseline():
    centered_score = compute_centered_boolq_score(2 / 3)

    assert centered_score == (2 / 3 - 0.62) / (1.0 - 0.62)


def test_compute_class_conditional_boolq_margin_means_splits_by_gold_label():
    data = [
        {'choices': ['No', 'Yes']},
        {'choices': ['Yes', 'No']},
        {'choices': ['No', 'Yes']},
        {'choices': ['Yes', 'No']},
    ]
    details = [
        {'index': 0, 'gold_idx': 1, 'choice_logps': [-4.0, -1.0]},
        {'index': 1, 'gold_idx': 0, 'choice_logps': [-1.0, -3.0]},
        {'index': 2, 'gold_idx': 1, 'choice_logps': [-2.5, -1.0]},
        {'index': 3, 'gold_idx': 1, 'choice_logps': [-2.0, -3.0]},
    ]

    means = compute_class_conditional_boolq_margin_means(details, data)

    assert means == {
        'mean_margin_yes_examples': 13 / 6,
        'mean_margin_no_examples': 1.0,
    }