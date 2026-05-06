import re


ALL_CHAT_EVAL_TASKS = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
CHATCORE_TASKS_WITHOUT_SPELLINGBEE = [task_name for task_name in ALL_CHAT_EVAL_TASKS if task_name != 'SpellingBee']
CHAT_EVAL_BASELINE_ACCURACIES = {
    'ARC-Easy': 0.25,
    'ARC-Challenge': 0.25,
    'MMLU': 0.25,
    'GSM8K': 0.0,
    'HumanEval': 0.0,
    'SpellingBee': 0.0,
}
ACCURACY_LINE_RE = re.compile(r'^(?P<task>.+?)\s+accuracy:\s+(?P<percent>\d+(?:\.\d+)?)%\s*$')


def compute_chatcore_metric(results, baseline_accuracies=None, all_tasks=None):
    baseline_accuracies = CHAT_EVAL_BASELINE_ACCURACIES if baseline_accuracies is None else baseline_accuracies
    all_tasks = ALL_CHAT_EVAL_TASKS if all_tasks is None else all_tasks
    metrics = {}

    def add_metric(metric_name, metric_tasks):
        if not metric_tasks or not all(task_name in results for task_name in metric_tasks):
            return

        centered_mean = 0.0
        for task_name in metric_tasks:
            acc = results[task_name]
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        metrics[metric_name] = centered_mean / len(metric_tasks)

    add_metric("ChatCORE metric", all_tasks)
    tasks_without_spellingbee = [task_name for task_name in all_tasks if task_name != 'SpellingBee']
    if tasks_without_spellingbee != list(all_tasks):
        add_metric("ChatCORE metric (without SpellingBee)", tasks_without_spellingbee)
    return metrics


def parse_accuracy_report(text):
    results = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = ACCURACY_LINE_RE.match(line)
        if not match:
            continue
        task_name = match.group("task")
        accuracy = float(match.group("percent")) / 100.0
        results[task_name] = accuracy
    return results