"""The eight fixed cross-vowel acoustic inputs of the historical study."""

import pandas as pd

VOICE = (
    "rel_i_lpc_F3_vs_e_abs_delta",
    "rel_i_mid_high_energy_ratio_vs_e_delta",
    "rel_i_b1_stable_vs_a_abs_delta",
    "rel_i_stable_score_vs_a_abs_delta",
    "rel_i_rolloff95_mean_vs_a_delta",
    "rel_i_h1_h3_db_vs_e_delta",
    "rel_i_cpps_full_vs_a_delta",
    "rel_i_lpc_F1_vs_a_abs_delta",
)

SOURCE_COLUMNS = (
    "i_lpc_F3", "e_lpc_F3", "i_mid_high_energy_ratio", "e_mid_high_energy_ratio",
    "i_b1_stable", "a_b1_stable", "i_stable_score", "a_stable_score",
    "i_rolloff95_mean", "a_rolloff95_mean", "i_h1_h3_db", "e_h1_h3_db",
    "i_cpps_full", "a_cpps_full", "i_lpc_F1", "a_lpc_F1",
)


def add_relative_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create historical signed/absolute differences from per-vowel measurements."""
    missing = sorted(set(SOURCE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing per-vowel columns: {', '.join(missing)}")
    out = frame.copy()
    for col in SOURCE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="raise")
    specs = (
        ("lpc_F3", "i", "e", True),
        ("mid_high_energy_ratio", "i", "e", False),
        ("b1_stable", "i", "a", True),
        ("stable_score", "i", "a", True),
        ("rolloff95_mean", "i", "a", False),
        ("h1_h3_db", "i", "e", False),
        ("cpps_full", "i", "a", False),
        ("lpc_F1", "i", "a", True),
    )
    for stem, right, left, absolute in specs:
        suffix = "abs_delta" if absolute else "delta"
        delta = out[f"{right}_{stem}"] - out[f"{left}_{stem}"]
        out[f"rel_{right}_{stem}_vs_{left}_{suffix}"] = delta.abs() if absolute else delta
    return out
