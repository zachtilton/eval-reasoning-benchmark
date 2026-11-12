"""Tests for src.analysis.findings_metrics."""

import pandas as pd

from src.analysis.findings_metrics import error_asymmetry_by_reference, majority_labels, reference_labels


def _db(labels_by_key):
    """Build a minimal scored DB: {(model, cond, frag): [run labels]}."""
    rows = []
    for (m, c, f), labels in labels_by_key.items():
        for lab in labels:
            rows.append({"model_family": m, "prompt_condition": c, "fragment_id": f,
                         "classification_output": lab, "error_flag": lab is None})
    return pd.DataFrame(rows)


def test_majority_labels_handles_unusable_runs():
    """2/3 agreement wins; a 1-1 split with an unusable run is 'none'."""
    db = _db({
        ("m", "zero_shot", 1): ["sound", "sound", "not_sound"],
        ("m", "zero_shot", 2): ["sound", "not_sound", None],
        ("m", "zero_shot", 3): ["not_sound", "not_sound", None],
    })
    out = majority_labels(db).set_index("fragment_id")["pred"].to_dict()
    assert out == {1: "sound", 2: "none", 3: "not_sound"}


def test_reference_labels_and_asymmetry():
    """Overrides apply per version; error rates are within-label."""
    gold = pd.DataFrame({"fragment_id": [1, 2, 3, 4], "expert_classification": ["sound", "sound", "not_sound", "not_sound"]})
    supplement = pd.DataFrame({"fragment_id": [3], "original_rating": ["sound"]})
    resolved = pd.DataFrame({"fragment_id": [4], "reviewer_classification": ["sound"]})
    ref = reference_labels(gold, supplement, resolved)
    assert ref.set_index("fragment_id").loc[3, "pre_revision"] == "sound"
    assert ref.set_index("fragment_id").loc[4, "triangulated"] == "sound"

    maj = pd.DataFrame({"model_family": "m", "prompt_condition": "zero_shot", "fragment_id": [1, 2, 3, 4],
                        "pred": ["sound", "not_sound", "sound", "sound"]})
    out = error_asymmetry_by_reference(maj, ref)
    assert out["locked"]["missed_flaws_pct"] == 100.0       # both gold not_sound called sound
    assert out["locked"]["wrongly_rejected_pct"] == 50.0    # 1 of 2 gold sound rejected


def test_unreviewed_bound_rule_of_three():
    """Bound = unreviewed x 3/reviewed per stratum; a census adds nothing."""
    from src.analysis.findings_metrics import _unreviewed_bound

    rev = pd.DataFrame({
        "stratum": ["incoherent_confident"] * 30 + ["ambiguous_sides_with_haiku"] * 60 + ["ambiguous_3way_split"] * 10,
        "your_rating": "coherent",
    })
    pop = {"coherent": 830, "incoherent": 60, "ambiguous": 100 + 10}
    luna = pd.DataFrame({"luna_verdict": ["coherent"] * 100 + ["ambiguous"] * 10})
    out = _unreviewed_bound(rev, pop, luna)
    assert out["incoherent"]["max_incoherent_unreviewed"] == 3.0          # 30 unreviewed x 3/30
    assert out["ambiguous_luna_coherent"]["max_incoherent_unreviewed"] == 2.0  # 40 unreviewed x 3/60
    assert out["ambiguous_luna_ambiguous"]["max_incoherent_unreviewed"] == 0.0  # census
    assert out["max_incoherent_unreviewed_pct_of_all_runs"] == 0.5             # 5 of 1,000
