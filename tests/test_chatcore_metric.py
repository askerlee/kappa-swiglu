import subprocess
import sys
from pathlib import Path

import pytest

from nanochat.chatcore import compute_chatcore_metric, parse_accuracy_report


ROOT = Path(__file__).resolve().parents[1]

SAMPLE_INPUT = """ARC-Easy accuracy: 33.59%
ARC-Challenge accuracy: 30.97%
MMLU accuracy: 30.31%
GSM8K accuracy: 2.81%
HumanEval accuracy: 2.44%
SpellingBee accuracy: 76.17%
"""


def test_parse_accuracy_report_parses_percent_lines():
    results = parse_accuracy_report(SAMPLE_INPUT)

    assert results == pytest.approx({
        "ARC-Easy": 0.3359,
        "ARC-Challenge": 0.3097,
        "MMLU": 0.3031,
        "GSM8K": 0.0281,
        "HumanEval": 0.0244,
        "SpellingBee": 0.7617,
    })


def test_compute_chatcore_metric_returns_both_metrics():
    results = parse_accuracy_report(SAMPLE_INPUT)

    metrics = compute_chatcore_metric(results)

    assert metrics == pytest.approx({
        "ChatCORE metric": 0.17985555555555555,
        "ChatCORE metric (without SpellingBee)": 0.06348666666666666,
    })


def test_chatcore_metric_script_reads_pasted_input_from_stdin():
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.chatcore_metric"],
        input=SAMPLE_INPUT,
        text=True,
        capture_output=True,
        cwd=ROOT,
        check=True,
    )

    assert completed.stdout == (
        "ChatCORE metric: 0.1799\n"
        "ChatCORE metric (without SpellingBee): 0.0635\n"
    )