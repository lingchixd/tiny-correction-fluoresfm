"""
Multiple-comparison correction (Holm).

Reads curve_persample.csv, runs a paired Wilcoxon signed-rank test of each tuning
tier against the baseline, then applies Holm-Bonferroni correction across tiers,
and reports raw and corrected p-values with significance markers.

Addresses Reviewer 3: multiple paired tests were reported without a
multiple-comparison correction. The corrected p-values reported here are used in
the Evaluation Protocol and the Table 1 caption of the paper; all significant
relationships remain significant after correction.

Usage: set CSV_PATH, then python holm_correction.py
CPU only, reads the csv produced by evaluate_progressive.py.
"""

import csv
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

# ============== edit here ==============
CSV_PATH = r"./results_progressive/curve_persample.csv"
BASELINE = "baseline"          # name of the baseline tier in the csv strategy column
METRICS  = ["psnr", "ssim"]    # one family of tests per metric
# Tiers compared against baseline (Table 1 order). Names must match the csv
# strategy column.
TIERS = ["pretrained", "out_only", "in_only", "in_out", "outer2", "outer4", "all"]
# =======================================

def load(csv_path):
    """Return data[strategy][metric] = {filename: value}, for pairing by filename."""
    data = defaultdict(lambda: defaultdict(dict))
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            s = row["strategy"]; fn = row["filename"]
            for m in METRICS:
                if m in row and row[m] not in ("", None):
                    data[s][m][fn] = float(row[m])
    return data

def paired(base_dict, tier_dict):
    """Align by filename; return the two paired arrays (common filenames only)."""
    common = [fn for fn in base_dict if fn in tier_dict]
    common.sort()
    b = np.array([base_dict[fn] for fn in common])
    t = np.array([tier_dict[fn] for fn in common])
    return b, t, common

def holm(pvals):
    """Holm-Bonferroni correction. Returns corrected p-values in the original order."""
    m = len(pvals)
    order = np.argsort(pvals)               # ascending
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)         # enforce monotonicity
        adj[idx] = min(running, 1.0)
    return adj

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

def main():
    data = load(CSV_PATH)
    print(f"[INFO] loaded {CSV_PATH}")
    print(f"[INFO] baseline='{BASELINE}'  tiers={TIERS}\n")

    for m in METRICS:
        print("=" * 78)
        print(f"metric: {m.upper()}")
        print("=" * 78)
        if BASELINE not in data or m not in data[BASELINE]:
            print(f"[WARN] baseline {m} not found, skipping"); continue
        base = data[BASELINE][m]

        raw_p, dirs, ns, names = [], [], [], []
        for tier in TIERS:
            if tier not in data or m not in data[tier]:
                print(f"[WARN] {tier} {m} not found, skipping"); continue
            b, t, common = paired(base, data[tier][m])
            if len(common) < 2:
                print(f"[WARN] {tier}: too few paired samples ({len(common)}), skipping"); continue
            try:
                stat, p = wilcoxon(t, b)
            except ValueError:
                # degenerate case, e.g. all differences zero
                p = 1.0
            direction = "up" if np.mean(t) > np.mean(b) else "down"
            raw_p.append(p); dirs.append(direction)
            ns.append(len(common)); names.append(tier)

        if not raw_p:
            continue
        adj_p = holm(np.array(raw_p))

        print(f"{'tier':12}{'n':>4}{'dir':>6}{'raw p':>14}{'Holm p':>14}{'raw':>7}{'Holm':>7}")
        print("-" * 78)
        for name, d, n, rp, ap in zip(names, dirs, ns, raw_p, adj_p):
            print(f"{name:12}{n:>4}{d:>6}{rp:>14.3e}{ap:>14.3e}{stars(rp):>7}{stars(ap):>7}")
        print()

    print("=" * 78)
    print("Holm p = corrected p-value; last two columns are raw/corrected markers")
    print("(*** p<0.001, ** p<0.01, * p<0.05). dir up/down = higher/lower than baseline.")

if __name__ == "__main__":
    main()
