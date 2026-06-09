from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAT_SFT = ROOT / "scripts" / "chat_sft.py"
def test_kappa_bias_lr_schedule_uses_total_iterations_helper_and_cli_scales():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "def get_kappa_bias_lr_scale(optimizer, step, num_iterations):" in source
    assert 'if group.get("name") == "kappa_bias" and group.get("kind") == "adamw":' in source
    assert 'end_scale=group.get("lr_scale_end", 1.0)' in source
    assert 'max_scale=group.get("lr_scale_max", 1.0)' in source


def test_kappa_bias_lr_schedule_wires_delay_and_warmup_cli_args():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'nolearn_iterations=group.get("lr_scale_nolearn_iterations", 0)' in source
    assert 'warmup_iterations=group.get("lr_scale_warmup_iterations", 1000)' in source


def test_chat_sft_uses_schedule_total_iterations_when_applying_kappa_bias_lr_scale():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "kappa_bias_schedule_total_iterations = get_kappa_bias_schedule_total_iterations(" in source
    assert 'kappa_bias_lr_scale = get_kappa_bias_lr_scale(' in source
    assert '        optimizer,' in source
    assert '        kappa_bias_schedule_total_iterations,' in source


def test_chat_sft_inherits_kappa_slope_max_scale_without_sft_warmup():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "def get_kappa_slope_max_scale" not in source
    assert 'moe_kappa_slope_max_scale = getattr(orig_model.config, "moe_kappa_slope_max_scale", 3.0)' in source
    assert 'dense_kappa_slope_max_scale = getattr(orig_model.config, "dense_kappa_slope_max_scale", 2.0)' in source
    assert 'orig_model.set_kappa_slope_max_scales(' in source


def test_chat_eval_task_names_default_to_all_tasks():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'chat_eval_task_names = ALL_CHAT_EVAL_TASKS if args.chat_eval_task_name is None else args.chat_eval_task_name.split(\'|\')' in source


def test_chat_eval_runs_only_on_last_step():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "if last_step:\n        model.eval()\n        orig_model.eval()\n        engine = Engine(orig_model, tokenizer)" in source
    assert "chat_eval_every" not in source


def test_final_checkpoint_is_saved_before_final_chat_eval():
    source = CHAT_SFT.read_text(encoding="utf-8")

    save_index = source.index("    # save checkpoint at the end of the run before the expensive final chat eval")
    chat_eval_index = source.index("    if last_step:\n        model.eval()\n        orig_model.eval()\n        engine = Engine(orig_model, tokenizer)")

    assert save_index < chat_eval_index


def test_kappa_bias_l2_anchor_cli_defaults_to_initial_and_wires_load_behavior():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'parser.add_argument("--kappa-bias-l2-anchor", type=str, choices=("initial", "zero"), default="zero"' in source
    assert '--use-kappa-swiglu-as-lr-scaler' not in source
    assert 'refresh_kappa_bias_references = args.kappa_bias_l2_anchor == "initial"' in source
    assert 'refresh_kappa_bias_references=refresh_kappa_bias_references' in source


def test_matrix_optimizer_inherits_from_base_checkpoint_unless_explicitly_set():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "matrix_optimizer_was_specified = arg_was_explicitly_set(sys.argv[1:], '--matrix-optimizer')" in source
    assert 'args.matrix_optimizer = meta.get("user_config", {}).get("matrix_optimizer", "muon")' in source
    assert 'print0(f"Inherited matrix_optimizer: {args.matrix_optimizer}")' in source
    assert 'print0(f"Specified matrix_optimizer: {args.matrix_optimizer}")' in source