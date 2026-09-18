import multiprocessing as _mp
import os
import traceback
from pathlib import Path

import hydra
import hydra.core.hydra_config
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig
from hydra.utils import instantiate

from tuning.objective import (
    create_objective,
    apply_overrides,
    build_overrides_from_params,
    get_method_search_space,
)
from utils.logging_utils import redirect_stdout_to_log
from utils.meta_writer import init_meta_json, write_meta_json

try:
    from transformers.utils import logging
    logging.set_verbosity_error()

    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").disabled = True
    logging.getLogger("httpcore").disabled = True
except ImportError:
    pass

N_SEEDS = 8


def _set_seed(seed):
    """Fix all random seeds for reproducibility."""
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # FP32 matmul for bit-identical eval (the search phase used TF32).
    torch.backends.cuda.matmul.allow_tf32 = False


def _run_seed(conn, seed_cfg, df, seed):
    """Run one FL training with a fixed seed in a subprocess."""
    try:
        _set_seed(seed)
        trainer = instantiate(seed_cfg.federated_method, _recursive_=False)
        trainer._init_federated(seed_cfg, df)
        results = trainer.begin_train()
        conn.send(("ok", results))
    except Exception as e:
        traceback.print_exc()
        conn.send(("error", str(e)))
    finally:
        conn.close()


def _build_multi_seed_tables(all_results, metric_name):
    """Build summary DataFrames (mean/std over seeds) for saving and printing."""
    n = len(all_results)
    summary_rows = []

    test_rows = []
    for r in all_results:
        row = {"test_loss": r.get("server_test_loss", float("nan"))}
        row.update(r.get("server_test_metrics", {}))
        test_rows.append(row)
    test_df = pd.DataFrame(test_rows)
    for col in test_df.columns:
        summary_rows.append(
            {
                "section": "server_global_test",
                "client": "",
                "metric": col,
                "mean": test_df[col].mean(),
                "std": test_df[col].std(ddof=0),
            }
        )

    sv_rows = [r.get("server_val_metrics", {}) for r in all_results]
    sv_df = pd.DataFrame(sv_rows)
    for col in sv_df.columns:
        summary_rows.append(
            {
                "section": "server_val",
                "client": "",
                "metric": col,
                "mean": sv_df[col].mean(),
                "std": sv_df[col].std(ddof=0),
            }
        )

    val_rows = [r.get("aggregated_val_metrics", {}) for r in all_results]
    val_df = pd.DataFrame(val_rows)
    for col in val_df.columns:
        summary_rows.append(
            {
                "section": "aggregated_client_val",
                "client": "",
                "metric": col,
                "mean": val_df[col].mean(),
                "std": val_df[col].std(ddof=0),
            }
        )

    best_vals = [r["best_val_tune_metric"] for r in all_results]
    best_rounds = [r.get("best_round", np.nan) for r in all_results]
    summary_rows.append(
        {
            "section": "val_tune_best_across_seeds",
            "client": "",
            "metric": metric_name,
            "mean": float(np.mean(best_vals)),
            "std": float(np.std(best_vals)),
        }
    )
    summary_rows.append(
        {
            "section": "best_round",
            "client": "",
            "metric": "round",
            "mean": float(np.nanmean(best_rounds)),
            "std": float(np.nanstd(best_rounds)),
        }
    )

    summary_df = pd.DataFrame(summary_rows)

    pc_rows = []

    def _emit_per_client(result_key, section_suffix):
        pcb_lists = [
            r.get(result_key) for r in all_results if r.get(result_key)
        ]
        if not pcb_lists:
            return
        n_local = max(len(pcb) for pcb in pcb_lists)
        for c in range(n_local):
            rounds = []
            for pcb in pcb_lists:
                info = pcb[c] if c < len(pcb) else None
                rounds.append(info["best_round"] if info else float("nan"))
            arr = np.array(rounds, dtype=float)
            pc_rows.append(
                {
                    "section": "per_client_best_round" + section_suffix,
                    "client": int(c),
                    "metric": "round",
                    "mean": float(np.nanmean(arr)),
                    "std": float(np.nanstd(arr)),
                }
            )

            after_count = sum(
                1
                for pcb in pcb_lists
                if (pcb[c] is not None and pcb[c].get("phase") == "after")
            )
            pc_rows.append(
                {
                    "section": "per_client_phase_after_frac" + section_suffix,
                    "client": int(c),
                    "metric": "frac_after",
                    "mean": after_count / len(pcb_lists),
                    "std": float("nan"),
                }
            )

            for key, section in (
                ("client_val", "per_client_val" + section_suffix),
                ("client_test", "per_client_test" + section_suffix),
            ):
                first_info = next(
                    (
                        pcb[c][key]
                        for pcb in pcb_lists
                        if pcb[c] is not None and pcb[c].get(key)
                    ),
                    None,
                )
                if first_info is None:
                    continue
                metric_keys = list(first_info["metrics"].keys())
                all_cols = ["loss"] + metric_keys
                for col in all_cols:
                    vals = []
                    for pcb in pcb_lists:
                        info = pcb[c] if c < len(pcb) else None
                        if info is None or info.get(key) is None:
                            vals.append(float("nan"))
                        elif col == "loss":
                            vals.append(info[key]["loss"])
                        else:
                            vals.append(
                                info[key]["metrics"].get(col, float("nan"))
                            )
                    arr = np.array(vals, dtype=float)
                    pc_rows.append(
                        {
                            "section": section,
                            "client": int(c),
                            "metric": col,
                            "mean": float(np.nanmean(arr)),
                            "std": float(np.nanstd(arr)),
                        }
                    )

    _emit_per_client("per_client_best", "")
    _emit_per_client("per_client_best_vk", "_vk")          # v_k (pure personal)
    _emit_per_client("per_client_best_global", "_global")  # w^t (raw global)
    _emit_per_client("per_client_best_wtk", "_wtk")        # w_k^t (local finetune)
    _emit_per_client("per_client_best_max3", "_max3")      # max(w^t, v_k, w_k^t)

    # max-of-3 winner breakdown: which of before=w^t / after=v_k / wtk=w_k^t produced the max, per seed.
    m3_lists = [
        r.get("per_client_best_max3")
        for r in all_results
        if r.get("per_client_best_max3")
    ]
    if m3_lists:
        n_m3 = max(len(pcb) for pcb in m3_lists)
        for c in range(n_m3):
            for ph in ("before", "after", "wtk"):
                cnt = sum(
                    1
                    for pcb in m3_lists
                    if pcb[c] is not None and pcb[c].get("phase") == ph
                )
                pc_rows.append(
                    {
                        "section": "per_client_winner_max3",
                        "client": int(c),
                        "metric": ph,
                        "mean": cnt / len(m3_lists),
                        "std": float("nan"),
                    }
                )

    per_client_df = pd.DataFrame(pc_rows) if pc_rows else None

    return summary_df, per_client_df, n


def _build_per_seed_tables(all_results):
    """Per-seed tables: one column per seed for each metric."""
    n = len(all_results)
    if n == 0:
        return {}
    seed_cols = [f"seed_{i}" for i in range(n)]

    test_rows = []
    for r in all_results:
        row = {"test_loss": r.get("server_test_loss", float("nan"))}
        row.update(r.get("server_test_metrics", {}))
        test_rows.append(row)
    server_test = pd.DataFrame(test_rows)
    if not server_test.empty and server_test.shape[1] > 0:
        server_test = server_test.T
        server_test.columns = seed_cols
        server_test.index.name = "metric"
    else:
        server_test = None

    sv_rows = [r.get("server_val_metrics", {}) for r in all_results]
    server_val = pd.DataFrame(sv_rows)
    if not server_val.empty and server_val.shape[1] > 0:
        server_val = server_val.T
        server_val.columns = seed_cols
        server_val.index.name = "metric"
    else:
        server_val = None

    def _build_per_client(result_key):
        n_local = 0
        for r in all_results:
            pcb = r.get(result_key)
            if pcb:
                n_local = max(n_local, len(pcb))
        out = {}
        if n_local == 0:
            return out
        for c in range(n_local):
            parts = []
            for key, split_label in (
                ("client_val", "validation"),
                ("client_test", "test"),
            ):
                first_info = next(
                    (
                        r[result_key][c][key]
                        for r in all_results
                        if r.get(result_key)
                        and c < len(r[result_key])
                        and r[result_key][c] is not None
                        and r[result_key][c].get(key)
                    ),
                    None,
                )
                if first_info is None:
                    continue
                metric_keys = list(first_info["metrics"].keys())
                all_cols = ["loss"] + metric_keys
                metric_rows = []
                for col in all_cols:
                    vals = []
                    for r in all_results:
                        pcb = r.get(result_key)
                        if (
                            not pcb
                            or c >= len(pcb)
                            or pcb[c] is None
                            or pcb[c].get(key) is None
                        ):
                            vals.append(float("nan"))
                            continue
                        info = pcb[c][key]
                        if col == "loss":
                            vals.append(info["loss"])
                        else:
                            vals.append(
                                info["metrics"].get(col, float("nan"))
                            )
                    metric_rows.append(
                        {"metric": col}
                        | {f"seed_{i}": v for i, v in enumerate(vals)}
                    )
                block = (
                    pd.DataFrame(metric_rows)
                    .set_index("metric")
                    .reindex(columns=seed_cols)
                    .reset_index()
                )
                block["split"] = split_label
                parts.append(block)
            if not parts:
                continue
            merged = pd.concat(parts, ignore_index=True)
            merged = merged[["split", "metric"] + seed_cols]
            out[int(c)] = merged
        return out

    per_client = _build_per_client("per_client_best")
    per_client_vk = _build_per_client("per_client_best_vk")
    per_client_global = _build_per_client("per_client_best_global")
    per_client_wtk = _build_per_client("per_client_best_wtk")
    per_client_max3 = _build_per_client("per_client_best_max3")

    return {
        "server_test": server_test,
        "server_val": server_val,
        "per_client": per_client,
        "per_client_vk": per_client_vk,
        "per_client_global": per_client_global,
        "per_client_wtk": per_client_wtk,
        "per_client_max3": per_client_max3,
    }


def _save_per_seed_tables(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tables = _build_per_seed_tables(all_results)
    if not tables:
        return
    for fname, key in (
        ("server_metrics_multiple_seeds_test.csv", "server_test"),
        ("server_metrics_multiple_seeds_val.csv", "server_val"),
    ):
        df = tables.get(key)
        if df is not None and not df.empty:
            path = os.path.join(output_dir, fname)
            df.to_csv(path)
            print(f"Saved per-seed table: {path}")

    client_dfs = tables.get("per_client") or {}
    for client_id, df in sorted(client_dfs.items()):
        if df is None or df.empty:
            continue
        fname = f"client_metrics_multiple_seeds_{client_id}.csv"
        path = os.path.join(output_dir, fname)
        df.to_csv(path, index=False)
        print(f"Saved per-seed table: {path}")

    for _key, _suffix in (
        ("per_client_vk", "_vk"),          # v_k (pure personal)
        ("per_client_global", "_global"),  # w^t (raw global)
        ("per_client_wtk", "_wtk"),        # w_k^t (local finetune)
        ("per_client_max3", "_max3"),      # max(w^t, v_k, w_k^t)
    ):
        for client_id, df in sorted((tables.get(_key) or {}).items()):
            if df is None or df.empty:
                continue
            path = os.path.join(
                output_dir, f"client_metrics_multiple_seeds_{client_id}{_suffix}.csv"
            )
            df.to_csv(path, index=False)
            print(f"Saved per-seed table: {path}")


def _print_seed_results(summary_df, per_client_df, n, metric_name):
    """Pretty-print multi-seed evaluation tables."""
    print(f"\nServer Global Test ({n} seeds):")
    st = summary_df[summary_df["section"] == "server_global_test"].drop(
        columns=["section", "client"]
    )
    print(st.to_string(index=False))

    sv = summary_df[summary_df["section"] == "server_val"]
    if not sv.empty:
        print(f"\nServer Validation ({n} seeds):")
        print(sv.drop(columns=["section", "client"]).to_string(index=False))

    av = summary_df[summary_df["section"] == "aggregated_client_val"]
    if not av.empty:
        print(f"\nAggregated Client Validation ({n} seeds):")
        print(av.drop(columns=["section", "client"]).to_string(index=False))

    if per_client_df is not None and not per_client_df.empty:
        for section, title in (
            (
                "per_client_val",
                "Per-Client Val (phase with better val vs other phase this round)",
            ),
            (
                "per_client_test",
                "Per-Client Test (matching chosen val phase, at best round)",
            ),
        ):
            sub = per_client_df[per_client_df["section"] == section]
            if sub.empty:
                continue
            pivot = sub.pivot_table(
                index="client",
                columns="metric",
                values=["mean", "std"],
                aggfunc="first",
            )
            print(f"\n{title} ({n} seeds):")
            print(pivot.to_string())

        ph = per_client_df[
            per_client_df["section"] == "per_client_phase_after_frac"
        ]
        if not ph.empty:
            print(f"\nPer-Client fraction 'after' won within-round ({n} seeds):")
            print(ph[["client", "mean"]].to_string(index=False))

        br = per_client_df[per_client_df["section"] == "per_client_best_round"]
        if not br.empty:
            print(f"\nPer-Client Best Round ({n} seeds):")
            print(
                br[["client", "mean", "std"]]
                .rename(columns={"mean": "mean_round", "std": "std_round"})
                .to_string(index=False)
            )

    tail = summary_df[
        summary_df["section"].isin(["val_tune_best_across_seeds", "best_round"])
    ]
    print("\nSummary:")
    print(tail.drop(columns=["section", "client"]).to_string(index=False))


def _save_multi_seed_tables(summary_df, per_client_df, output_dir):
    """Write aggregated multi-seed results next to Hydra ``output.txt``."""
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "multi_seed_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved multi-seed summary: {summary_path}")

    if per_client_df is not None and not per_client_df.empty:
        pc_path = os.path.join(output_dir, "multi_seed_per_client.csv")
        per_client_df.to_csv(pc_path, index=False)
        print(f"Saved per-client summary: {pc_path}")

        for section, suffix in (
            ("per_client_val", "val"),
            ("per_client_test", "test"),
        ):
            sub = per_client_df[per_client_df["section"] == section]
            if sub.empty:
                continue
            wide = sub.pivot_table(
                index="client",
                columns="metric",
                values=["mean", "std"],
                aggfunc="first",
            )
            wide_path = os.path.join(
                output_dir, f"multi_seed_per_client_{suffix}_wide.csv"
            )
            wide.to_csv(wide_path)
            print(f"Saved per-client ({suffix}) wide table: {wide_path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def tune(cfg: DictConfig):
    redirect_stdout_to_log()

    # cuDNN autotune + TF32 for the search; the eval phase re-pins determinism.
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(4)

    df = instantiate(cfg.train_dataset, cfg=cfg, mode="train", _recursive_=False)

    tuning_cfg = cfg.tuning
    metric_name = tuning_cfg.metric

    if tuning_cfg.enable_pruning:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=tuning_cfg.n_startup_trials,
            n_warmup_steps=tuning_cfg.n_warmup_steps,
        )
    else:
        pruner = optuna.pruners.NopPruner()
        print("[tune.py] Pruning DISABLED — all trials will run to completion.", flush=True)

    # Use an exhaustive GridSampler if the method exposes SEARCH_GRID; else default TPE.
    _method_space = get_method_search_space(cfg)
    _search_grid = getattr(_method_space, "SEARCH_GRID", None)
    if _search_grid:
        _grid = {k: list(v) for k, v in _search_grid.items()}
        grid_n_trials = 1
        for _vals in _grid.values():
            grid_n_trials *= len(_vals)
        sampler = optuna.samplers.GridSampler(_grid)
        pruner = optuna.pruners.NopPruner()
        print(
            f"[tune.py] GRID search over {list(_grid.keys())}: {grid_n_trials} "
            f"points, pruning forced OFF.",
            flush=True,
        )
    else:
        sampler = None  # default TPESampler
        grid_n_trials = None

    # Auto-derive the storage DB path from study_name when storage is unset.
    storage = tuning_cfg.storage
    study_name = tuning_cfg.study_name
    if not storage and isinstance(study_name, str) and study_name.strip():
        try:
            from utils.meta_writer import canonical_dataset_name, federated_method_name

            repo_root = Path(__file__).resolve().parents[1]
            db_path = (
                repo_root / "tuning_studies"
                / canonical_dataset_name(cfg)
                / federated_method_name(cfg)
                / f"{study_name.strip()}.db"
            )
            db_path.parent.mkdir(parents=True, exist_ok=True)
            storage = f"sqlite:///{db_path.as_posix()}"
            print(f"[tune.py] auto-derived tuning.storage: {storage}", flush=True)
        except Exception as exc:  # bookkeeping must not fail the run
            print(f"[tune.py] WARNING: could not auto-derive storage: {exc}", flush=True)

    study = optuna.create_study(
        direction=tuning_cfg.direction,
        study_name=tuning_cfg.study_name,
        storage=storage,
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True,
    )

    objective = create_objective(cfg, df, metric_name=metric_name)

    try:
        run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        init_meta_json(cfg, Path(run_dir), storage=storage)
    except Exception as exc:  # bookkeeping must not fail the run
        print(f"[tune.py] WARNING: could not init meta.json: {exc}")

    n_trials = grid_n_trials if grid_n_trials is not None else tuning_cfg.n_trials
    # Resume-aware: n_trials is the total useful-trials target; the missing count is derived from the DB on restart.
    _states = [t.state for t in study.trials]
    done = sum(
        s in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        for s in _states
    )
    if _states:
        prior_failed = sum(s == optuna.trial.TrialState.FAIL for s in _states)
        stale = sum(s == optuna.trial.TrialState.RUNNING for s in _states)
        print(
            f"[tune.py] Resuming study '{study.study_name}': {done} useful "
            f"trials in DB ({prior_failed} failed, {stale} stale-running "
            f"ignored) → running {max(0, n_trials - done)} more to reach "
            f"the n_trials={n_trials} target.",
            flush=True,
        )
    n_trials = max(0, n_trials - done)

    if n_trials > 0:
        # A crashed trial is recorded as FAILED and the study continues (Ctrl+C still works).
        study.optimize(objective, n_trials=n_trials, catch=(Exception,))

    try:
        run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        meta_path = write_meta_json(cfg, study, Path(run_dir), storage=storage)
        print(f"[tune.py] Wrote run manifest: {meta_path}")
    except Exception as exc:  # bookkeeping must not fail the run
        print(f"[tune.py] WARNING: could not write meta.json: {exc}")

    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)

    total = len(study.trials)
    completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    pruned = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    )
    failed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.FAIL
    ]
    failed = len(failed_trials)
    print(
        f"\nTrials: {total} total, {completed} completed, "
        f"{pruned} pruned, {failed} failed"
    )
    for t in failed_trials:
        print(f"  FAILED trial #{t.number}: params={t.params}")
    if total > 0 and failed / total > 0.2:
        print(
            f"\n[tune.py] WARNING: {failed}/{total} trials FAILED (>20%) — "
            f"the sampler explored a crippled search space; fix the cause "
            f"(see tracebacks above) and rerun this study.",
            flush=True,
        )
    if completed == 0:
        print(
            "\n[tune.py] No completed trials — skipping best-trial report and "
            "multi-seed evaluation.",
            flush=True,
        )
        return

    best = study.best_trial
    print(f"\nBest trial #{best.number}:")
    print(f"  Validation {metric_name}: {best.value:.4f}")

    print("\nBest hyperparameters:")
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print(f"MULTI-SEED EVALUATION ({N_SEEDS} seeds)")
    print("=" * 60)

    best_params = study.best_params
    overrides = build_overrides_from_params(cfg, best_params)
    overrides["federated_params.print_client_metrics"] = False

    print(f"\nConfig overrides: {overrides}")

    all_results = []
    for seed in range(N_SEEDS):
        print(f"\n{'─' * 40}")
        print(f"Seed {seed}/{N_SEEDS - 1}")
        print(f"{'─' * 40}")

        seed_cfg = apply_overrides(cfg, overrides)

        reader, writer = _mp.Pipe(duplex=False)
        p = _mp.Process(target=_run_seed, args=(writer, seed_cfg, df, seed))
        p.start()
        writer.close()
        try:
            status, value = reader.recv()
        except EOFError:
            p.join()
            print(f"  Subprocess crashed (exit code {p.exitcode})")
            continue
        p.join()

        if status == "ok":
            all_results.append(value)
            bv = value["best_val_tune_metric"]
            br = value.get("best_round", "?")
            print(f"  Result: best_val={bv:.4f}, best_round={br}")
        else:
            print(f"  FAILED: {value}")

    if all_results:
        print("\n" + "=" * 60)
        print("AGGREGATED RESULTS")
        print("=" * 60)
        summary_df, per_client_df, n_seeds = _build_multi_seed_tables(
            all_results, metric_name
        )
        _print_seed_results(summary_df, per_client_df, n_seeds, metric_name)
        out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        _save_multi_seed_tables(summary_df, per_client_df, out_dir)
        _save_per_seed_tables(all_results, out_dir)

        # Mirror tables into results/.../tables/ so they are git-tracked and portable.
        try:
            from utils.meta_writer import run_results_dir

            tables_dir = str(run_results_dir(cfg) / "tables")
            _save_multi_seed_tables(summary_df, per_client_df, tables_dir)
            _save_per_seed_tables(all_results, tables_dir)
            print(f"[tune.py] Wrote report tables: {tables_dir}")
        except Exception as exc:
            print(f"[tune.py] WARNING: could not write report tables to results/: {exc}")
    else:
        print("\nNo successful seed runs!")


if __name__ == "__main__":
    tune()
