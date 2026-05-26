import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "average_result_csv.py"
SPEC = importlib.util.spec_from_file_location("average_result_csv", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_result_csv(path: Path, arc_acc: float, arc_centered: float, core: float, core_no_boolq: float):
    path.write_text(
        "\n".join(
            [
                "Task, Accuracy, Centered",
                f"arc, {arc_acc:.6f}, {arc_centered:.6f}",
                "boolq, 0.500000, 0.100000",
                f"CORE, , {core:.6f}",
                f"CORE (no boolq), , {core_no_boolq:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_collects_base_seed_groups_and_writes_average(tmp_path: Path):
    write_result_csv(
        tmp_path / "exp32-d12-gbias-lin-au-s24_base_011303.csv",
        arc_acc=0.100000,
        arc_centered=0.200000,
        core=0.150000,
        core_no_boolq=0.200000,
    )
    write_result_csv(
        tmp_path / "exp32-d12-gbias-lin-au-s26_base_011303.csv",
        arc_acc=0.200000,
        arc_centered=0.400000,
        core=0.250000,
        core_no_boolq=0.300000,
    )
    write_result_csv(
        tmp_path / "exp32-d12-gbias-lin-au-s28_base_011303.csv",
        arc_acc=0.300000,
        arc_centered=0.600000,
        core=0.350000,
        core_no_boolq=0.400000,
    )
    write_result_csv(
        tmp_path / "znoort-exp32-d12-au-s24_base_011297.csv",
        arc_acc=0.800000,
        arc_centered=0.700000,
        core=0.600000,
        core_no_boolq=0.500000,
    )
    (tmp_path / "exp32-d12-chat-results.txt").write_text("ignore me\n", encoding="utf-8")
    (tmp_path / "exp32-d12-s24_chat_000520.csv").write_text("ignore me\n", encoding="utf-8")

    groups = MODULE.collect_method_files(tmp_path)

    assert set(groups) == {"exp32-d12-gbias-lin-au", "znoort-exp32-d12-au"}

    written_paths = MODULE.write_average_files(tmp_path, groups)

    assert written_paths == [tmp_path / "exp32-d12-gbias-lin-au_base_avg.csv"]

    header, rows = MODULE.load_csv_rows(written_paths[0])
    assert header == ["Task", "Accuracy", "Centered"]
    assert rows == [
        ["arc", "0.200000", "0.400000"],
        ["boolq", "0.500000", "0.100000"],
        ["CORE", "", "0.250000"],
        ["CORE (no boolq)", "", "0.300000"],
    ]


def test_write_average_files_supports_explicit_output_dir(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "averaged"
    input_dir.mkdir()

    write_result_csv(
        input_dir / "znoort-exp32-d12-au-s24_base_011297.csv",
        arc_acc=0.800000,
        arc_centered=0.700000,
        core=0.600000,
        core_no_boolq=0.500000,
    )
    write_result_csv(
        input_dir / "znoort-exp32-d12-au-s26_base_011297.csv",
        arc_acc=0.600000,
        arc_centered=0.500000,
        core=0.400000,
        core_no_boolq=0.300000,
    )

    groups = MODULE.collect_method_files(input_dir)

    written_paths = MODULE.write_average_files(output_dir, groups)

    assert written_paths == [output_dir / "znoort-exp32-d12-au_base_avg.csv"]
    assert not (input_dir / "znoort-exp32-d12-au_base_avg.csv").exists()