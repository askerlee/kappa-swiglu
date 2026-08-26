import ast
from pathlib import Path

from datasets import Dataset

from tasks import tulu3


ROOT = Path(__file__).resolve().parents[1]


def test_tulu3_persona_if_loads_focused_training_dataset(monkeypatch):
    messages = [
        {"role": "user", "content": "Answer using exactly three sentences."},
        {"role": "assistant", "content": "First. Second. Third."},
    ]
    load_calls = []

    def fake_load_dataset(path, split):
        load_calls.append((path, split))
        return Dataset.from_dict({"messages": [messages]})

    monkeypatch.setattr(tulu3, "load_dataset", fake_load_dataset)

    task = tulu3.Tulu3SFTPersonaIF(split="train")

    assert load_calls == [("allenai/tulu-3-sft-personas-instruction-following", "train")]
    assert len(task) == 1
    assert task[0] == {"messages": messages}


def test_tulu3_english_filter_checks_full_conversation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    english_messages = [
        {"role": "user", "content": "Analyze the meaning of this song."},
        {"role": "assistant", "content": "The song describes memory and loss."},
    ]
    telugu_messages = [
        {"role": "user", "content": "ఈ పాటను విశ్లేషించండి."},
        {"role": "assistant", "content": "ఈ పాట జీవితం గురించి చెబుతుంది."},
    ]
    mixed_messages = [
        {"role": "user", "content": "Write a detailed analysis of this song and its themes."},
        {"role": "assistant", "content": "ఈ పాట జీవితం గురించి చెబుతుంది."},
    ]

    def fake_load_dataset(path, split):
        return Dataset.from_dict({"messages": [english_messages, telugu_messages, mixed_messages]})

    monkeypatch.setattr(tulu3, "load_dataset", fake_load_dataset)

    task = tulu3.Tulu3SFTMixture(split="train", english_only=True)

    assert len(task) == 1
    assert task[0] == {"messages": english_messages}


def test_tulu3_english_filter_uses_allocated_slurm_cpus(monkeypatch):
    filter_kwargs = {}

    class FakeDataset:
        def filter(self, predicate, **kwargs):
            filter_kwargs.update(kwargs)
            return self

        def shuffle(self, seed):
            return self

        def __len__(self):
            return 0

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setattr(tulu3, "load_dataset", lambda path, split: FakeDataset())

    tulu3.Tulu3SFTMixture(split="train", english_only=True)

    assert filter_kwargs["num_proc"] == 8
    assert filter_kwargs["desc"] == "Filtering non-English Tulu conversations"


def test_tulu3_english_filter_divides_pbs_cpus_between_local_ranks(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setenv("PBS_NCPUS", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")

    assert tulu3.get_filter_num_proc() == 8


def test_tulu3_english_filter_uses_local_cpu_affinity(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("PBS_NCPUS", raising=False)
    monkeypatch.delenv("NCPUS", raising=False)
    monkeypatch.delenv("PBS_NP", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setattr(tulu3.os, "sched_getaffinity", lambda process_id: {0, 1, 2, 3})

    assert tulu3.get_filter_num_proc() == 4


def test_training_scripts_enable_english_only_tulu_data():
    for script_name in ("base_train_mix.py", "chat_sft.py"):
        tree = ast.parse((ROOT / "scripts" / script_name).read_text())
        tulu_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"Tulu3SFTMixture", "Tulu3SFTPersonaIF"}
        ]

        assert len(tulu_calls) == 2
        for call in tulu_calls:
            english_only = next(
                keyword.value
                for keyword in call.keywords
                if keyword.arg == "english_only"
            )
            assert isinstance(english_only, ast.Constant)
            assert english_only.value is True


def test_pbs_jobs_request_fourteen_cpus_per_gpu():
    expected_resources = {
        "base_train_mix.pbs": "select=1:ncpus=14:ngpus=1",
        "base_train_mix2u.pbs": "select=1:ncpus=28:ngpus=2",
        "base_train_mix4u.pbs": "select=1:ncpus=56:ngpus=4",
        "chat_train.pbs": "select=1:ncpus=14:ngpus=1",
        "chat_train2u.pbs": "select=1:ncpus=28:ngpus=2",
    }

    for script_name, resources in expected_resources.items():
        assert f"#PBS -l {resources}" in (ROOT / script_name).read_text()