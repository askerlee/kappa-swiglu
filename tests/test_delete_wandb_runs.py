from types import SimpleNamespace
from unittest.mock import Mock

import wandb

from scripts.delete_wandb_runs import main


def test_delete_oldest_runs_requires_execute(monkeypatch, capsys):
    runs = [
        SimpleNamespace(id=str(index), name=f"run-{index}", created_at=str(index), delete=Mock())
        for index in range(3)
    ]
    list_runs = Mock(return_value=iter(runs))
    monkeypatch.setattr(wandb, "Api", lambda: SimpleNamespace(runs=list_runs))

    main(["team/project", "2"])
    list_runs.assert_called_once_with("team/project", order="+created_at")
    assert "Dry run" in capsys.readouterr().out
    assert all(not run.delete.called for run in runs)

    list_runs.return_value = iter(runs)
    main(["team/project", "oldest-runs", "2"])
    assert all(not run.delete.called for run in runs)

    list_runs.return_value = iter(runs)
    main(["team/project", "2", "--execute"])
    assert [run.delete.call_count for run in runs] == [1, 1, 0]


def test_delete_largest_files_across_runs_requires_execute(monkeypatch, capsys):
    files = [
        SimpleNamespace(name=f"file-{index}", size=size, delete=Mock())
        for index, size in enumerate((10, 50, 30, 20))
    ]
    runs = [
        SimpleNamespace(id="first", name="first run", created_at="2026-01-01T00:00:00", files=lambda: iter(files[:2])),
        SimpleNamespace(id="second", name="second run", created_at="2026-02-01T00:00:00", files=lambda: iter(files[2:])),
    ]
    list_runs = Mock(return_value=iter(runs))
    monkeypatch.setattr(wandb, "Api", lambda: SimpleNamespace(runs=list_runs))

    main(["team/project", "largest-files", "2"])
    list_runs.assert_called_once_with("team/project")
    preview = capsys.readouterr().out
    assert "50 bytes  2026-01-01T00:00:00  first  first run  file-1" in preview
    assert "30 bytes  2026-02-01T00:00:00  second  second run  file-2" in preview
    assert "Dry run" in preview
    assert all(not file.delete.called for file in files)

    list_runs.return_value = iter(runs)
    main(["team/project", "largest-files", "2", "--execute"])
    assert [file.delete.call_count for file in files] == [0, 1, 1, 0]