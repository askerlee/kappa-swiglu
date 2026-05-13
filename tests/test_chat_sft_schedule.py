from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAT_SFT = ROOT / "scripts" / "chat_sft.py"
def test_gate_proj_bias_lr_schedule_uses_total_iterations_helper_and_cli_scales():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "def get_gate_proj_bias_lr_scale(step, progress):" in source
    assert "gate_proj_bias_schedule_total_iterations = get_gate_proj_bias_schedule_total_iterations(step, progress)" in source
    assert "end_scale=args.gate_proj_bias_lr_final_scale" in source
    assert "max_scale=args.gate_proj_bias_lr_max_scale" in source


def test_gate_proj_bias_lr_schedule_wires_delay_and_warmup_cli_args():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "nolearn_iterations=args.gate_proj_bias_delay_start_iterations" in source
    assert "warmup_iterations=args.gate_proj_bias_lr_warmup_iterations" in source


def test_chat_eval_task_names_default_to_all_tasks():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'chat_eval_task_names = ALL_CHAT_EVAL_TASKS if args.chat_eval_task_name is None else args.chat_eval_task_name.split(\'|\')' in source


def test_chat_eval_runs_only_on_last_step():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "if last_step:\n        model.eval()\n        engine = Engine(orig_model, tokenizer)" in source
    assert "chat_eval_every" not in source


def test_final_checkpoint_is_saved_before_final_chat_eval():
    source = CHAT_SFT.read_text(encoding="utf-8")

    save_index = source.index("    # save checkpoint at the end of the run before the expensive final chat eval")
    chat_eval_index = source.index("    if last_step:\n        model.eval()\n        engine = Engine(orig_model, tokenizer)")

    assert save_index < chat_eval_index


def test_gate_proj_bias_l2_anchor_cli_defaults_to_initial_and_wires_load_behavior():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'parser.add_argument("--exp-gate-proj-bias-l2-anchor", type=str, choices=("initial", "zero"), default="zero"' in source
    assert "parser.add_argument(\"--use-gate-proj-bias-as-lr-scaler\", type=str2bool, nargs='?', const=True, default=False" in source
    assert 'refresh_gate_proj_bias_references = args.exp_gate_proj_bias_l2_anchor == "initial"' in source
    assert 'refresh_gate_proj_bias_references=refresh_gate_proj_bias_references' in source


def test_matrix_optimizer_inherits_from_base_checkpoint_unless_explicitly_set():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "matrix_optimizer_was_specified = arg_was_explicitly_set(sys.argv[1:], '--matrix-optimizer')" in source
    assert 'args.matrix_optimizer = meta.get("user_config", {}).get("matrix_optimizer", "muon")' in source
    assert 'print0(f"Inherited matrix_optimizer: {args.matrix_optimizer}")' in source
    assert 'print0(f"Specified matrix_optimizer: {args.matrix_optimizer}")' in source