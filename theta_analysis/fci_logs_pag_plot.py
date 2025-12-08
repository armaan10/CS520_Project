# fci_logs_pag_plot.py
import os
import numpy as np
import pandas as pd

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import gsq
from causallearn.utils.GraphUtils import GraphUtils

# --------------------
# Settings & paths
# --------------------
PATH_IN = "/Users/sdarmora/theta_project/data/log_data/filter_data/cleaned_joblog_20181231_with_exit.csv"
OUTDIR  = "/Users/sdarmora/theta_project/results"
os.makedirs(OUTDIR, exist_ok=True)

ALPHA    = 0.01
Q_BINS   = 4
SAMPLE_N = 10000

# --------------------
# Load & prep data
# --------------------
df = pd.read_csv(PATH_IN, low_memory=False)
if len(df) > SAMPLE_N:
    df = df.sample(n=SAMPLE_N, random_state=42)

vars_cont = ["NODES_REQUESTED","CORES_REQUESTED","NODES_USED","CORES_USED",
             "RUNTIME_SECONDS","WALLTIME_SECONDS"]

# derive JOB_DURATION_SEC if needed (correct timedelta path)
if "JOB_DURATION_SEC" not in df.columns and "JOB_DURATION" in df.columns:
    td = pd.to_timedelta(df["JOB_DURATION"], errors="coerce")
    df["JOB_DURATION_SEC"] = td.dt.total_seconds()
if "JOB_DURATION_SEC" in df.columns:
    vars_cont.append("JOB_DURATION_SEC")

# numeric coercion & drop NAs
for c in vars_cont:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
keep = [c for c in vars_cont if c in df.columns]
df = df.dropna(subset=keep)

# discrete outcome
df["EXIT_FAIL"] = (pd.to_numeric(df["EXIT_STATUS"], errors="coerce").fillna(0) != 0).astype(int)

# Discretize continuous vars to Q_BINS for discrete CI test
disc_arrays = []
for c in keep:
    try:
        b = pd.qcut(df[c], q=Q_BINS, labels=False, duplicates="drop")
    except ValueError:
        b = pd.cut(df[c].rank(method="average"), bins=Q_BINS, labels=False, include_lowest=True)
    disc_arrays.append(b.astype(int).to_numpy())

var_order = keep + ["EXIT_FAIL"]
data = np.column_stack(disc_arrays + [df["EXIT_FAIL"].to_numpy(dtype=int)])

# --------------------
# 1) Plain FCI-PAG
# --------------------
G_plain, _ = fci(data, gsq, alpha=ALPHA)

png_plain   = os.path.join(OUTDIR, "fci_logs_q4alpha001_pag.png")
legend_path = os.path.join(OUTDIR, "fci_logs_q4alpha001_legend.txt")

pdot_plain = GraphUtils.to_pydot(G_plain)
pdot_plain.write_png(png_plain)

with open(legend_path, "w") as f:
    for i, name in enumerate(var_order, start=1):
        f.write(f"X{i} = {name}\n")

# --------------------
# 2) Sink-style PAG via DOT post-processing (remove edges out of EXIT_FAIL)
# --------------------
exit_node_name = f"X{len(var_order)}"  # EXIT_FAIL is last in var_order

import copy
pdot_sink = copy.deepcopy(pdot_plain)

# Collect and remove all edges with source EXIT_FAIL
to_remove = []
for e in pdot_sink.get_edges():
    try:
        src = e.get_source()
        dst = e.get_destination()
    except Exception:
        continue
    if src == exit_node_name:
        to_remove.append((src, dst))

for src, dst in to_remove:
    # del_edge may have multiple parallel edges; wrap in try
    try:
        pdot_sink.del_edge(src, dst)
    except Exception:
        pass

png_sink = os.path.join(OUTDIR, "fci_logs_q4alpha001_pag_sink.png")
pdot_sink.write_png(png_sink)

print("Wrote:", png_plain)
print("Wrote:", png_sink)
print("Legend:", legend_path)

