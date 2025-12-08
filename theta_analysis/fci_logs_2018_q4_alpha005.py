# fci_logs_2018_q4.py
import os, numpy as np, pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import gsq  # G^2 discrete CI test

PATH_IN = "/Users/sdarmora/theta_project/data/log_data/filter_data/cleaned_joblog_20181231_with_exit.csv"
OUTDIR  = "/Users/sdarmora/theta_project/results"
os.makedirs(OUTDIR, exist_ok=True)
OUT_TXT = os.path.join(OUTDIR, "fci_logs_2018_q4alpha005_edges.txt")

# --- load + sample ---
df = pd.read_csv(PATH_IN, low_memory=False)
if len(df) > 10000:
    df = df.sample(n=10000, random_state=42)

# --- continuous timing vars from logs ---
vars_cont = [
    "NODES_REQUESTED","CORES_REQUESTED",
    "NODES_USED","CORES_USED",
    "RUNTIME_SECONDS","WALLTIME_SECONDS"
]

# Derive JOB_DURATION_SEC if needed
if "JOB_DURATION_SEC" not in df.columns and "JOB_DURATION" in df.columns:
    def to_sec(x):
        try:
            if isinstance(x, str) and "day" in x: return pd.to_timedelta(x).total_seconds()
            return float(x)
        except: return np.nan
    df["JOB_DURATION_SEC"] = df["JOB_DURATION"].apply(to_sec)

if "JOB_DURATION_SEC" in df.columns:
    vars_cont.append("JOB_DURATION_SEC")

# coerce numerics & drop rows with missing in these vars
for c in vars_cont:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
keep_cols = [c for c in vars_cont if c in df.columns]
df = df.dropna(subset=keep_cols)

# --- discrete outcome: EXIT_FAIL ---
if "EXIT_STATUS" not in df.columns:
    raise SystemExit("File must contain EXIT_STATUS.")
df["EXIT_FAIL"] = (pd.to_numeric(df["EXIT_STATUS"], errors="coerce").fillna(0) != 0).astype(int)

# --- discretize continuous vars to 4 quantile bins (0..3) for discrete CI test ---
disc_cols = []
disc_arrays = []
for c in keep_cols:
    try:
        bins = pd.qcut(df[c], q=4, labels=False, duplicates="drop")
    except ValueError:
        # fallback if not enough unique values
        bins = pd.cut(df[c].rank(method="average"), bins=4, labels=False, include_lowest=True)
    bins = bins.astype(int)
    disc_cols.append(c)
    disc_arrays.append(bins.to_numpy())

# build all-discrete data matrix for gsq: [discretized timing vars..., EXIT_FAIL]
var_order = disc_cols + ["EXIT_FAIL"]
data = np.column_stack(disc_arrays + [df["EXIT_FAIL"].to_numpy(dtype=int)])

# --- run FCI (alpha=0.01) with G^2 test ---
alpha = 0.05
G, _ = fci(data, gsq, alpha=alpha)

# --- write the PAG edges ---
with open(OUT_TXT, "w") as f:
    f.write(f"Variables (order): {var_order}\nAlpha: {alpha}\n\nPAG edges:\n{G}\n")

print("Done.")
print("Variables (order):", var_order)
print("Alpha:", alpha)
print("Edges written to:", OUT_TXT)

