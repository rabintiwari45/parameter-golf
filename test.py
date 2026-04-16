import torch
import numpy as np
import logging
import io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_input):
    """
    Accepts either:
      - A file path (str)                  → loads from disk
      - An OrderedDict / dict              → already loaded, use directly
      - Raw bytes                          → wraps in BytesIO then loads
      - A BytesIO / file-like object       → loads directly
    """
    # Case 1: Already a dict/OrderedDict in memory — use it directly,
    # no torch.load() needed (this is what caused the 'seek' error)
    if isinstance(checkpoint_input, dict):
        logger.info("Checkpoint is already loaded in memory as a dict — skipping torch.load().")
        return checkpoint_input

    # Case 2: File path string — standard load from disk
    if isinstance(checkpoint_input, str):
        logger.info(f"Loading checkpoint from file path: '{checkpoint_input}'")
        return torch.load(checkpoint_input, map_location="cpu")

    # Case 3: Raw bytes — wrap in BytesIO buffer first, then load
    if isinstance(checkpoint_input, bytes):
        logger.info("Checkpoint is raw bytes — wrapping in BytesIO buffer and loading.")
        buffer = io.BytesIO(checkpoint_input)
        return torch.load(buffer, map_location="cpu")

    # Case 4: Already a seekable file-like object (BytesIO, open file, etc.)
    if hasattr(checkpoint_input, "read"):
        logger.info("Checkpoint is a file-like object — loading directly.")
        return torch.load(checkpoint_input, map_location="cpu")

    raise TypeError(
        f"Unsupported checkpoint type: {type(checkpoint_input)}. "
        "Expected a file path (str), dict, bytes, or file-like object."
    )


def _is_state_dict(d: dict) -> bool:
    """
    A raw state dict maps string keys directly to Tensors.
    A wrapped checkpoint has mixed values (epoch int, loss float, etc.)
    """
    return all(isinstance(v, torch.Tensor) for v in d.values())


def _extract_state_dict(checkpoint: dict) -> dict:
    """
    Handles wrapped checkpoints saved as:
        torch.save({"model": model.state_dict(), "optimizer": ..., "epoch": ...})
    Tries common key names and raises a clear error if none match.
    """
    for key in ("model", "model_state_dict", "state_dict", "net", "network"):
        if key in checkpoint:
            logger.info(f"Wrapped checkpoint detected — extracting state dict from key: '{key}'")
            return checkpoint[key]

    raise KeyError(
        f"Could not find a state dict inside the checkpoint. "
        f"Available keys: {list(checkpoint.keys())}. "
        "Pass the correct key name manually."
    )


def profile_outliers_from_checkpoint(checkpoint_input, threshold_percentile=99.9):
    """
    Accepts a file path, an already-loaded OrderedDict, raw bytes, or a
    file-like object and profiles every weight matrix for outliers that
    could degrade INT4 quantization quality.
    """

    # ------------------------------------------------------------------ #
    #  Step 1 — Resolve checkpoint to a state dict                        #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)

    checkpoint = load_checkpoint(checkpoint_input)

    if not _is_state_dict(checkpoint):
        state_dict = _extract_state_dict(checkpoint)
    else:
        state_dict = checkpoint

    matrix_keys = [
        k for k, v in state_dict.items()
        if isinstance(v, torch.Tensor) and v.dim() >= 2
    ]

    logger.info(f"Total keys in checkpoint : {len(state_dict)}")
    logger.info(
        f"Weight matrices (dim ≥ 2): {len(matrix_keys)}  "
        f"← Only these will be profiled; biases and scalars are skipped."
    )
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    #  Step 2 — Profile each weight matrix                                 #
    # ------------------------------------------------------------------ #
    outlier_report = {}
    total_layers   = len(matrix_keys)

    for idx, name in enumerate(matrix_keys, start=1):
        param = state_dict[name]

        logger.info(f"[{idx}/{total_layers}] Profiling layer: '{name}'")
        logger.info(
            f"  Shape: {list(param.shape)}  |  "
            f"Total weights: {param.numel():,}"
        )

        w = param.detach().float()

        # --- Absolute Maximum ---
        abs_max = w.abs().max().item()
        logger.info(
            f"  Absolute max weight : {abs_max:.6f}  "
            f"← The single largest weight value in this layer."
        )

        # --- 99.9th Percentile Threshold ---
        p999 = np.percentile(w.abs().cpu().numpy(), threshold_percentile)
        logger.info(
            f"  {threshold_percentile}th percentile : {p999:.6f}  "
            f"← Weights above this are outliers "
            f"(top {100 - threshold_percentile:.1f}% of distribution)."
        )

        # --- Kurtosis ---
        kurtosis_val = (
            torch.mean((w - w.mean()) ** 4) / (torch.var(w) ** 2 + 1e-8)
        ).item()
        kurtosis_label = (
            "⚠  Very heavy tails — high quantization risk"  if kurtosis_val > 10 else
            "⚠  Moderately heavy tails — watch this layer"  if kurtosis_val > 5  else
            "✓  Near-normal distribution — low risk"
        )
        logger.info(
            f"  Kurtosis            : {kurtosis_val:.2f}  (normal ≈ 3.0)  "
            f"← {kurtosis_label}"
        )

        # --- Dynamic Range ---
        mean_abs      = w.abs().mean().item()
        dynamic_range = abs_max / (mean_abs + 1e-8)
        range_label   = (
            "⚠  Very wide — INT4 will likely clip small weights"  if dynamic_range > 100 else
            "⚠  Moderate — some precision loss expected"          if dynamic_range > 20  else
            "✓  Compact range — INT4 should handle this well"
        )
        logger.info(
            f"  Dynamic range       : {dynamic_range:.1f}x  "
            f"(max {abs_max:.4f} / mean {mean_abs:.4f})  "
            f"← {range_label}"
        )

        # --- Outlier Ratio ---
        outlier_ratio = (w.abs() > p999).float().mean().item()
        logger.info(
            f"  Outlier ratio       : {outlier_ratio * 100:.3f}%  "
            f"← Expected ~{100 - threshold_percentile:.1f}% for a healthy distribution."
        )

        # --- Risk & Recommendation ---
        is_high_risk   = kurtosis_val > 10 or dynamic_range > 100
        is_medium_risk = kurtosis_val > 5  or dynamic_range > 20
        if is_high_risk:
            rec = "🔴 HIGH RISK   — Keep in INT8 or apply SmoothQuant before INT4."
        elif is_medium_risk:
            rec = "🟡 MEDIUM RISK — Use smaller group size (64) or GPTQ compensation."
        else:
            rec = "🟢 LOW RISK    — Safe for standard INT4 with group size 128."
        logger.info(f"  Recommendation      : {rec}")
        logger.info("")

        outlier_report[name] = {
            "abs_max"                    : abs_max,
            f"p{threshold_percentile}"   : p999,
            "kurtosis"                   : kurtosis_val,
            "dynamic_range"              : dynamic_range,
            "outlier_ratio"              : outlier_ratio,
            "risk"                       : "high" if is_high_risk else "medium" if is_medium_risk else "low",
        }

    # ------------------------------------------------------------------ #
    #  Step 3 — Summary                                                    #
    # ------------------------------------------------------------------ #
    high_risk   = [n for n, r in outlier_report.items() if r["risk"] == "high"]
    medium_risk = [n for n, r in outlier_report.items() if r["risk"] == "medium"]
    low_risk    = [n for n, r in outlier_report.items() if r["risk"] == "low"]

    logger.info("=" * 60)
    logger.info("PROFILING COMPLETE — SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  🔴 High risk layers   : {len(high_risk)}   ← Must handle carefully (INT8 / SmoothQuant)")
    logger.info(f"  🟡 Medium risk layers : {len(medium_risk)}   ← Benefit from smaller group size or GPTQ")
    logger.info(f"  🟢 Low risk layers    : {len(low_risk)}   ← Safe for standard INT4 quantization")

    if high_risk:
        logger.info("")
        logger.info("  High risk layers (check these first):")
        for n in high_risk:
            r = outlier_report[n]
            logger.info(
                f"    • {n}  "
                f"(kurtosis={r['kurtosis']:.1f}, "
                f"dynamic_range={r['dynamic_range']:.1f}x)"
            )

    logger.info("=" * 60)
    return outlier_report


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # Works for all four cases:

    # 1. File path (most common)
    report = profile_outliers_from_checkpoint("final_model_backup.pt")

    # 2. Already-loaded OrderedDict (fixes the original 'seek' error)
    # already_loaded = torch.load("final_model_backup.pt", map_location="cpu")
    # report = profile_outliers_from_checkpoint(already_loaded)

    # 3. Raw bytes (e.g. fetched from a remote store)
    # with open("final_model_backup.pt", "rb") as f:
    #     raw_bytes = f.read()
    # report = profile_outliers_from_checkpoint(raw_bytes)