"""Experiment execution and benchmarking runner for Models A, B, C, and D.

Implements:
1. Sequential execution of all four VAE experiments.
2. Strict reproducibility contract (seeding before model construction and DataLoader generation).
3. Rigorous checkpoint verification (reloading and bit-for-bit weight comparison).
4. GPU VRAM tracking and cleanup between runs.
5. Standardized output structure under results/experiments/ and checkpoints/.
6. Accurate runtime and memory benchmarking.
"""

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config import ProjectConfig
from src.data.dataset_loader import get_dataloaders
from src.models.vae import VAE, build_vae
from src.training.checkpointing import load_checkpoint, save_checkpoint, save_history
from src.training.trainer import Trainer, set_seed


def verify_saved_checkpoint(
    checkpoint_path: Path,
    model: VAE,
    model_type: str,
    expected_epoch: int,
    expected_val_loss: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Verifies that the checkpoint exists, reloads correctly, and matches expected state.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model: The trained VAE instance (already restored to best state).
        model_type: Model identifier ('A', 'B', 'C', 'D').
        expected_epoch: Best epoch returned by Trainer.fit().
        expected_val_loss: Best validation loss returned by Trainer.fit().
        device: Device on which verification is run.
        
    Returns:
        Dict of verification results.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Verification failed: Checkpoint file does not exist at {checkpoint_path}")

    # Build a fresh, clean model instance to load into
    verification_model = build_vae(model_type).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model=verification_model, device=device)

    # 1. Verify metadata matches
    saved_epoch = checkpoint["epoch"]
    saved_val_loss = checkpoint["val_loss"]

    if saved_epoch != expected_epoch:
        raise ValueError(
            f"Checkpoint epoch mismatch for Model {model_type}: "
            f"saved={saved_epoch} vs expected={expected_epoch}"
        )

    if not np.isclose(saved_val_loss, expected_val_loss, atol=1e-5):
        raise ValueError(
            f"Checkpoint validation loss mismatch for Model {model_type}: "
            f"saved={saved_val_loss} vs expected={expected_val_loss}"
        )

    # 2. Verify model weights match bit-for-bit
    for name, param in model.named_parameters():
        v_param = dict(verification_model.named_parameters())[name]
        torch.testing.assert_close(
            param,
            v_param,
            rtol=0.0,
            atol=0.0,
            msg=f"Model {model_type} restored weight mismatch for parameter {name}",
        )

    # 3. For Models C and D, explicitly verify learnable prior parameters exist in checkpoint
    has_prior_centers = False
    has_prior_log_stds = False
    if model_type in ["C", "D"]:
        state_dict = checkpoint["model_state_dict"]
        has_prior_centers = "prior.centers" in state_dict
        has_prior_log_stds = "prior.log_stds" in state_dict
        if not (has_prior_centers and has_prior_log_stds):
            raise ValueError(
                f"Model {model_type} checkpoint is missing learnable prior parameters! "
                f"centers_in_ckpt={has_prior_centers}, log_stds_in_ckpt={has_prior_log_stds}"
            )

    # 4. For Model B, verify fixed prior buffers match a fresh Model B prior exactly
    fixed_prior_verified = True
    if model_type == "B":
        fresh_b = build_vae("B").to(device)
        state_dict = checkpoint["model_state_dict"]

        if "prior.centers" not in state_dict or "prior.log_stds" not in state_dict or "prior.weights" not in state_dict:
            raise ValueError("Model B checkpoint is missing fixed prior buffers!")

        # Compare model's restored buffers against fresh Model B prior buffers (exact equality)
        torch.testing.assert_close(model.prior.centers, fresh_b.prior.centers, rtol=0.0, atol=0.0)
        torch.testing.assert_close(model.prior.log_stds, fresh_b.prior.log_stds, rtol=0.0, atol=0.0)
        torch.testing.assert_close(model.prior.weights, fresh_b.prior.weights, rtol=0.0, atol=0.0)

        # Compare checkpoint's saved buffers against fresh Model B prior buffers
        torch.testing.assert_close(state_dict["prior.centers"], fresh_b.prior.centers, rtol=0.0, atol=0.0)
        torch.testing.assert_close(state_dict["prior.log_stds"], fresh_b.prior.log_stds, rtol=0.0, atol=0.0)
        torch.testing.assert_close(state_dict["prior.weights"], fresh_b.prior.weights, rtol=0.0, atol=0.0)

        # Compare reloaded verification_model buffers against fresh Model B prior buffers
        torch.testing.assert_close(verification_model.prior.centers, fresh_b.prior.centers, rtol=0.0, atol=0.0)
        torch.testing.assert_close(verification_model.prior.log_stds, fresh_b.prior.log_stds, rtol=0.0, atol=0.0)
        torch.testing.assert_close(verification_model.prior.weights, fresh_b.prior.weights, rtol=0.0, atol=0.0)

    return {
        "verified": True,
        "checkpoint_path": str(checkpoint_path),
        "saved_epoch": saved_epoch,
        "saved_val_loss": saved_val_loss,
        "learnable_prior_verified": has_prior_centers and has_prior_log_stds if model_type in ["C", "D"] else True,
        "fixed_prior_verified": fixed_prior_verified,
    }


def run_single_experiment(
    model_type: str,
    config: Optional[ProjectConfig] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    device: Optional[Union[str, torch.device]] = None,
    max_epochs: Optional[int] = None,
    patience: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Runs a single end-to-end VAE training experiment.
    
    Args:
        model_type: One of 'A', 'B', 'C', 'D'.
        config: Project configuration.
        train_loader: Training DataLoader. If None, created with config seed.
        val_loader: Validation DataLoader.
        device: Target device (defaults to config.device).
        max_epochs: Maximum epochs.
        patience: Early stopping patience.
        verbose: Verbose printing flag.
        
    Returns:
        Dict containing experiment summary, paths, and verified metrics.
    """
    config = config or ProjectConfig()
    model_type = model_type.upper()

    dev = torch.device(device or config.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested or configured, but torch.cuda.is_available() is False!")

    # 1. Enforce reproducibility: seed BEFORE model creation
    set_seed(config.training.model_seed)

    # Validate explicit arguments
    effective_max_epochs = config.training.max_epochs if max_epochs is None else max_epochs
    effective_patience = config.training.early_stopping_patience if patience is None else patience

    # 2. Construct Model
    model = build_vae(model_type, config).to(dev)

    # 3. Create DataLoaders if not provided (with training seed)
    if train_loader is None or val_loader is None:
        train_loader, val_loader, _, _ = get_dataloaders(
            batch_size=config.training.batch_size,
            seed=config.training.model_seed,
        )

    # Output directory paths
    exp_dir = config.paths.experiments_dir / f"model_{model_type}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.paths.checkpoints_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"model_{model_type}_best.pt"
    history_json_path = exp_dir / "history.json"
    history_csv_path = exp_dir / "history.csv"
    metadata_json_path = exp_dir / "metadata.json"

    # Reset GPU memory stats if using CUDA
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.empty_cache()

    trainer = Trainer(
        model=model,
        model_type=model_type,
        config=config,
        device=dev,
    )

    t_start = time.perf_counter()

    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=effective_max_epochs,
        patience=effective_patience,
        verbose=verbose,
        save_checkpoints=True,
        checkpoint_dir=checkpoint_dir,
        history_dir=exp_dir,
    )

    elapsed_time = time.perf_counter() - t_start

    # Record GPU peak memory
    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
    if dev.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(dev) / (1024 ** 2)

    # 4. Rigorous Checkpoint Verification
    verification = verify_saved_checkpoint(
        checkpoint_path=checkpoint_path,
        model=result.model,
        model_type=model_type,
        expected_epoch=result.best_epoch,
        expected_val_loss=result.best_val_loss,
        device=dev,
    )

    # 5. Save model metadata
    metadata = {
        "model_type": model_type,
        "best_epoch": result.best_epoch,
        "total_epochs": result.total_epochs,
        "stopped_early": result.stopped_early,
        "best_val_loss": float(result.best_val_loss),
        "best_val_recon": float(result.best_val_recon),
        "best_val_kl": float(result.best_val_kl),
        "elapsed_time_sec": round(elapsed_time, 2),
        "peak_allocated_mb": round(peak_allocated_mb, 2),
        "peak_reserved_mb": round(peak_reserved_mb, 2),
        "device": str(dev),
        "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU",
        "training_seed": config.training.model_seed,
        "val_eval_seed": config.training.val_eval_seed,
        "lr": config.training.lr,
        "batch_size": config.training.batch_size,
        "max_epochs": int(effective_max_epochs),
        "patience": int(effective_patience),
        "likelihood_sigma": config.model.likelihood_sigma,
        "checkpoint_path": str(checkpoint_path),
        "verification": verification,
    }

    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save dedicated copy to experiment directory: results/experiments/model_{type}/
    save_history(
        history=result.history,
        json_path=history_json_path,
        csv_path=history_csv_path,
        extra_summary=metadata,
    )

    # Also save standard copy to results/histories/ for compatibility
    save_history(
        history=result.history,
        json_path=config.paths.histories_dir / f"model_{model_type}_history.json",
        csv_path=config.paths.histories_dir / f"model_{model_type}_history.csv",
        extra_summary=metadata,
    )

    # Clean up GPU memory
    del model, trainer
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    return metadata


def run_all_experiments(
    models: List[str] = ("A", "B", "C", "D"),
    config: Optional[ProjectConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Sequentially executes training experiments for all requested models.
    
    Args:
        models: List of model identifiers to run sequentially.
        config: Project configuration.
        device: Target device.
        verbose: Verbose printing flag.
        
    Returns:
        Summary dict containing all individual model results.
    """
    config = config or ProjectConfig()
    dev = torch.device(device or config.device)

    print(f"\n=======================================================")
    print(f"Launching Sequential VAE Experiments: Models {list(models)}")
    print(f"Device: {dev} ({torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'CPU'})")
    print(f"Config: Max Epochs={config.training.max_epochs}, Patience={config.training.early_stopping_patience}, Batch Size={config.training.batch_size}")
    print(f"=======================================================\n")

    summary_records = []
    total_start = time.perf_counter()

    for m_type in models:
        # Load fresh DataLoaders with reproducibility seed
        train_loader, val_loader, _, _ = get_dataloaders(
            batch_size=config.training.batch_size,
            seed=config.training.model_seed,
        )

        res = run_single_experiment(
            model_type=m_type,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=dev,
            verbose=verbose,
        )
        summary_records.append(res)

    total_time = time.perf_counter() - total_start

    # Save overall experiment summary
    exp_summary = {
        "total_experiments_time_sec": round(total_time, 2),
        "total_experiments_time_min": round(total_time / 60.0, 2),
        "device": str(dev),
        "models": summary_records,
    }

    config.paths.experiments_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = config.paths.experiments_dir / "experiment_summary.json"
    summary_csv_path = config.paths.experiments_dir / "experiment_summary.csv"

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(exp_summary, f, indent=2)

    # Write summary CSV
    csv_fields = [
        "model_type",
        "best_epoch",
        "total_epochs",
        "stopped_early",
        "best_val_loss",
        "best_val_recon",
        "best_val_kl",
        "elapsed_time_sec",
        "peak_allocated_mb",
        "peak_reserved_mb",
    ]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in summary_records:
            writer.writerow({k: r[k] for k in csv_fields})

    print(f"\n=======================================================")
    print(f"All {len(models)} Experiments Completed in {total_time:.2f}s ({total_time / 60.0:.2f} min)")
    print(f"Summary saved to: {summary_json_path}")
    print(f"=======================================================\n")

    return exp_summary


def benchmark_training_runtime(
    config: Optional[ProjectConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    warmup_steps: int = 5,
    measure_steps: int = 20,
) -> Dict[str, Any]:
    """Empirically benchmarks complete step times, epoch times, and VRAM usage on the active device.
    
    Runs a non-destructive benchmark using Model C (the most computationally involved model
    with 8 learnable GMM components and MC KL) on the actual training DataLoader.
    
    Separately measures:
    1. DataLoader iteration time (retrieving batch from CPU iterator).
    2. Host-to-device transfer time (batch['x'].to(device) with CUDA synchronization).
    3. Model training / validation computation time (forward, loss, backward, step).
    4. Complete training batch time (iteration + transfer + computation).
    5. Complete validation batch time (iteration + transfer + computation).
    6. Checkpoint save time to disk.
    
    Args:
        config: Project configuration.
        device: Target device (defaults to config.device).
        warmup_steps: Number of initial steps to discard for CUDA warm-up.
        measure_steps: Number of steps to average over.
        
    Returns:
        Dict containing measured timings, GPU stats, and projected runtimes.
    """
    import tempfile
    config = config or ProjectConfig()
    dev = torch.device(device or config.device)

    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for benchmark, but torch.cuda.is_available() is False!")

    set_seed(config.training.model_seed)
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=config.training.batch_size,
        seed=config.training.model_seed,
    )

    n_train_batches = len(train_loader)  # 27 batches (26 * 256 + 64 = 6720)
    n_val_batches = len(val_loader)      # 6 batches (5 * 256 + 160 = 1440)

    # Benchmark with Model C (learnable 8-GMM with logsumexp)
    model = build_vae("C", config).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    val_generator = torch.Generator(device=dev.type)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 1. Warm-up Training Steps
    model.train()
    train_iter = iter(train_loader)
    
    for _ in range(warmup_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        x = batch["x"].to(dev)
        optimizer.zero_grad()
        out = model(x)
        loss = model.compute_loss(x, out)
        loss.total_loss.backward()
        optimizer.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # 2. Timed Training Steps (measuring iteration, transfer, computation, and complete batch)
    train_iter_times: List[float] = []
    train_h2d_times: List[float] = []
    train_comp_times: List[float] = []
    train_batch_complete_times: List[float] = []

    for _ in range(measure_steps):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_batch_start = time.perf_counter()

        # Step 2a: DataLoader iteration (retrieving next batch from CPU iterator)
        t_iter_start = time.perf_counter()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        t_iter_end = time.perf_counter()
        train_iter_times.append(t_iter_end - t_iter_start)

        # Step 2b: Host-to-device transfer
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_h2d_start = time.perf_counter()
        x = batch["x"].to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_h2d_end = time.perf_counter()
        train_h2d_times.append(t_h2d_end - t_h2d_start)

        # Step 2c: Model training computation (zero_grad, forward, loss, backward, step)
        t_comp_start = time.perf_counter()
        optimizer.zero_grad()
        out = model(x)
        loss = model.compute_loss(x, out)
        loss.total_loss.backward()
        optimizer.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_comp_end = time.perf_counter()
        train_comp_times.append(t_comp_end - t_comp_start)

        # Complete batch time
        t_batch_end = time.perf_counter()
        train_batch_complete_times.append(t_batch_end - t_batch_start)

    mean_train_iter_sec = float(np.mean(train_iter_times))
    mean_train_h2d_sec = float(np.mean(train_h2d_times))
    mean_train_comp_sec = float(np.mean(train_comp_times))
    mean_train_batch_complete_sec = float(np.mean(train_batch_complete_times))

    # 3. Warm-up Validation Steps
    model.eval()
    val_iter = iter(val_loader)
    val_generator.manual_seed(config.training.val_eval_seed)

    with torch.no_grad():
        for _ in range(min(warmup_steps, len(val_loader))):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                batch = next(val_iter)
            x = batch["x"].to(dev)
            out = model(x, generator=val_generator)
            loss = model.compute_loss(x, out)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # 4. Timed Validation Steps (measuring iteration, transfer, computation, and complete batch)
    val_iter_times: List[float] = []
    val_h2d_times: List[float] = []
    val_comp_times: List[float] = []
    val_batch_complete_times: List[float] = []

    with torch.no_grad():
        for _ in range(measure_steps):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_batch_start = time.perf_counter()

            # Step 4a: DataLoader iteration
            t_iter_start = time.perf_counter()
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                batch = next(val_iter)
            t_iter_end = time.perf_counter()
            val_iter_times.append(t_iter_end - t_iter_start)

            # Step 4b: Host-to-device transfer
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_h2d_start = time.perf_counter()
            x = batch["x"].to(dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_h2d_end = time.perf_counter()
            val_h2d_times.append(t_h2d_end - t_h2d_start)

            # Step 4c: Model validation computation
            t_comp_start = time.perf_counter()
            out = model(x, generator=val_generator)
            loss = model.compute_loss(x, out)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_comp_end = time.perf_counter()
            val_comp_times.append(t_comp_end - t_comp_start)

            # Complete batch time
            t_batch_end = time.perf_counter()
            val_batch_complete_times.append(t_batch_end - t_batch_start)

    mean_val_iter_sec = float(np.mean(val_iter_times))
    mean_val_h2d_sec = float(np.mean(val_h2d_times))
    mean_val_comp_sec = float(np.mean(val_comp_times))
    mean_val_batch_complete_sec = float(np.mean(val_batch_complete_times))

    # 5. Measure Checkpoint Save Time
    ckpt_save_times: List[float] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_ckpt = Path(tmp_dir) / "bench_ckpt.pt"
        for _ in range(3):
            t_s = time.perf_counter()
            save_checkpoint(
                checkpoint_path=tmp_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=1,
                val_loss=10.0,
                val_recon_loss=5.0,
                val_kl_loss=5.0,
                model_type="C",
            )
            t_e = time.perf_counter()
            ckpt_save_times.append(t_e - t_s)
    mean_ckpt_save_sec = float(np.mean(ckpt_save_times))

    # 6. Calculate Complete Epoch Duration from Complete Batch Timings
    train_epoch_time_sec = n_train_batches * mean_train_batch_complete_sec
    val_epoch_time_sec = n_val_batches * mean_val_batch_complete_sec
    single_epoch_total_sec = train_epoch_time_sec + val_epoch_time_sec

    # 7. Measure GPU VRAM
    peak_alloc_mb = 0.0
    peak_res_mb = 0.0
    if dev.type == "cuda":
        peak_alloc_mb = float(torch.cuda.max_memory_allocated(dev) / (1024 ** 2))
        peak_res_mb = float(torch.cuda.max_memory_reserved(dev) / (1024 ** 2))

    # Cleanup
    del model, optimizer
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # 8. Project Run Durations with Clearly Stated Assumptions
    # - Minimum: Early stopping triggers at earliest possible epoch (15 patience epochs after epoch 1 = 16 epochs, 1 save).
    # - Typical Low: Convergence within 15 epochs + 15 patience epochs = 30 epochs, 5 saves.
    # - Typical High: Convergence within 45 epochs + 15 patience epochs = 60 epochs, 10 saves.
    # - Maximum: Full 100 epochs reached without early stopping, 15 saves.
    min_epochs = 16
    min_saves = 1
    typ_low_epochs = 30
    typ_low_saves = 5
    typ_high_epochs = 60
    typ_high_saves = 10
    max_epochs = config.training.max_epochs  # 100
    max_saves = 15

    single_model_min_sec = (min_epochs * single_epoch_total_sec) + (min_saves * mean_ckpt_save_sec)
    single_model_typ_low_sec = (typ_low_epochs * single_epoch_total_sec) + (typ_low_saves * mean_ckpt_save_sec)
    single_model_typ_high_sec = (typ_high_epochs * single_epoch_total_sec) + (typ_high_saves * mean_ckpt_save_sec)
    single_model_max_sec = (max_epochs * single_epoch_total_sec) + (max_saves * mean_ckpt_save_sec)

    all_4_min_sec = 4 * single_model_min_sec
    all_4_typ_low_sec = 4 * single_model_typ_low_sec
    all_4_typ_high_sec = 4 * single_model_typ_high_sec
    all_4_max_sec = 4 * single_model_max_sec

    benchmark_report = {
        "device": str(dev),
        "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU",
        "n_train_batches": n_train_batches,
        "n_val_batches": n_val_batches,
        "batch_size": config.training.batch_size,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "measured_metrics": {
            "training_batch_breakdown_ms": {
                "dataloader_iteration_ms": round(mean_train_iter_sec * 1000.0, 3),
                "host_to_device_transfer_ms": round(mean_train_h2d_sec * 1000.0, 3),
                "model_computation_ms": round(mean_train_comp_sec * 1000.0, 3),
                "complete_training_batch_ms": round(mean_train_batch_complete_sec * 1000.0, 3),
            },
            "validation_batch_breakdown_ms": {
                "dataloader_iteration_ms": round(mean_val_iter_sec * 1000.0, 3),
                "host_to_device_transfer_ms": round(mean_val_h2d_sec * 1000.0, 3),
                "model_computation_ms": round(mean_val_comp_sec * 1000.0, 3),
                "complete_validation_batch_ms": round(mean_val_batch_complete_sec * 1000.0, 3),
            },
            "checkpoint_save_ms": round(mean_ckpt_save_sec * 1000.0, 3),
            "complete_epoch_time_sec": round(single_epoch_total_sec, 4),
            "epoch_breakdown_sec": {
                "training_epoch_sec": round(train_epoch_time_sec, 4),
                "validation_epoch_sec": round(val_epoch_time_sec, 4),
            },
            "gpu_peak_allocated_mb": round(peak_alloc_mb, 2),
            "gpu_peak_reserved_mb": round(peak_res_mb, 2),
            # Compatibility fields
            "mean_train_step_ms": round(mean_train_batch_complete_sec * 1000.0, 3),
            "mean_val_step_ms": round(mean_val_batch_complete_sec * 1000.0, 3),
            "single_epoch_total_sec": round(single_epoch_total_sec, 4),
        },
        "projection_assumptions": {
            "min_scenario": "16 epochs (early stopping triggers after 1 best epoch + 15 non-improving epochs), 1 checkpoint save",
            "typ_low_scenario": "30 epochs (15 improving epochs + 15 non-improving epochs), 5 checkpoint saves",
            "typ_high_scenario": "60 epochs (45 improving epochs + 15 non-improving epochs), 10 checkpoint saves",
            "max_scenario": "100 epochs (full run without early stopping), 15 checkpoint saves",
            "epoch_formula": f"{n_train_batches} * complete_training_batch_sec + {n_val_batches} * complete_validation_batch_sec",
            "total_formula": "epochs * complete_epoch_sec + saves * checkpoint_save_sec",
        },
        "projections": {
            "single_model_min_16ep_sec": round(single_model_min_sec, 2),
            "single_model_typ_30ep_sec": round(single_model_typ_low_sec, 2),
            "single_model_typ_60ep_sec": round(single_model_typ_high_sec, 2),
            "single_model_max_100ep_sec": round(single_model_max_sec, 2),
            "all_4_models_min_16ep_sec": round(all_4_min_sec, 2),
            "all_4_models_typ_range_sec": (
                round(all_4_typ_low_sec, 2),
                round(all_4_typ_high_sec, 2),
            ),
            "all_4_models_max_100ep_sec": round(all_4_max_sec, 2),
        },
    }

    return benchmark_report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run VAE training experiments or benchmark.")
    parser.add_argument("--benchmark", action="store_true", help="Run runtime benchmark only.")
    parser.add_argument("--train", action="store_true", help="Run full sequential training for Models A, B, C, D.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. cuda:0, cpu).")
    args = parser.parse_args()

    if args.train:
        summary = run_all_experiments(device=args.device)
    else:
        rep = benchmark_training_runtime(device=args.device)
        print("Benchmark complete:")
        print(json.dumps(rep, indent=2))
