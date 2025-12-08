# fci_logs_pag_plot_labeled.py
import os, copy
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import gsq
from causallearn.utils.GraphUtils import GraphUtils

# ---------- helpers ----------
def _nm(x):
    """Normalize a pydot node/edge endpoint to a plain string name."""
    return str(x).strip('"') if x is not None else ""

def label_and_optionally_prune(pdot_graph, var_order, drop_isolates=False):
    """
    Return a deep-copied pydot graph with human-readable labels.
    Optionally drop isolated nodes (no incident edges) for cleaner slides.
    """
    g = copy.deepcopy(pdot_graph)
    name_map = {f"X{i+1}": var_order[i] for i in range(len(var_order))}

    # set readable labels while keeping internal node ids X1..Xn
    for n in g.get_nodes():
        nm = _nm(n.get_name())
        if nm in name_map:
            n.set_label(name_map[nm])

    if drop_isolates:
        connected = set()
        for e in g.get_edges():
            connected.add(_nm(e.get_source()))
            connected.add(_nm(e.get_destination()))
        # remove nodes not touching any edge
        for n in list(g.get_nodes()):
            if _nm(n.get_name()) not in connected:
                g.del_node(n.get_name())
    return g

# ---------- paths / settings ----------
PATH_IN = "/Users/sdarmora/theta_project/data/log_data/filter_data/cleaned_joblog_20181231_with_exit.csv"
OUTDIR  = "/Users/sdarmora/theta_project/results"
os.makedirs(OUTDIR, exist_ok=True)

ALPHA    = 0.01      # CI test alpha
Q_BINS   = 4         # discretization bins
SAMPLE_N = 10000     # keep runs quick

# ---------- load & prep ----------
df = pd.read_csv(PATH_IN, low_memory=False)
if len(df) > SAMPLE_N:
    df = df.sample(n=SAMPLE_N, random_state=42)

vars_cont = [
    "NODES_REQUESTED","CORES_REQUESTED","NODES_USED","CORES_USED",
    "RUNTIME_SECONDS","WALLTIME_SECONDS"
]

# derive JOB_DURATION_SEC if needed
if "JOB_DURATION_SEC" not in df.columns and "JOB_DURATION" in df.columns:
    td = pd.to_timedelta(df["JOB_DURATION"], errors="coerce")
    df["JOB_DURATION_SEC"] = td.dt.total_seconds()
if "JOB_DURATION_SEC" in df.columns:
    vars_cont.append("JOB_DURATION_SEC")

# numeric coercion & NA drop
for c in vars_cont:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
keep = [c for c in vars_cont if c in df.columns]
df = df.dropna(subset=keep)

# binary outcome
df["EXIT_FAIL"] = (pd.to_numeric(df["EXIT_STATUS"], errors="coerce").fillna(0) != 0).astype(int)

# discretize continuous vars for discrete CI test
disc_arrays = []
for c in keep:
    try:
        b = pd.qcut(df[c], q=Q_BINS, labels=False, duplicates="drop")
    except ValueError:
        b = pd.cut(df[c].rank(method="average"), bins=Q_BINS, labels=False, include_lowest=True)
    disc_arrays.append(b.astype(int).to_numpy())

var_order = keep + ["EXIT_FAIL"]  # X1..Xn mapping (legend)
data = np.column_stack(disc_arrays + [df["EXIT_FAIL"].to_numpy(dtype=int)])

# ---------- run FCI ----------
G, _ = fci(data, gsq, alpha=ALPHA)
pdot = GraphUtils.to_pydot(G)

# ---------- 1) plain PAG, labeled ----------
png_plain = os.path.join(OUTDIR, "fci_logs_pag_plain_labeled.png")
label_and_optionally_prune(pdot, var_order, drop_isolates=False).write_png(png_plain)

# ---------- 2) sink-style (remove edges out of EXIT_FAIL), labeled ----------
exit_node = f"X{len(var_order)}"  # EXIT_FAIL is last in var_order
pdot_sink = copy.deepcopy(pdot)

to_remove = []
for e in pdot_sink.get_edges():
    src = _nm(e.get_source()); dst = _nm(e.get_destination())
    if src == exit_node:
        to_remove.append((src, dst))
for src, dst in to_remove:
    try:
        pdot_sink.del_edge(src, dst)
    except Exception:
        pass

png_sink = os.path.join(OUTDIR, "fci_logs_pag_sink_labeled.png")
label_and_optionally_prune(pdot_sink, var_order, drop_isolates=False).write_png(png_sink)

# ---------- 3) sink-style, labeled, pruned (hide isolates) ----------
png_sink_pruned = os.path.join(OUTDIR, "fci_logs_pag_sink_labeled_pruned.png")
label_and_optionally_prune(pdot_sink, var_order, drop_isolates=True).write_png(png_sink_pruned)

# ---------- legend ----------
legend_path = os.path.join(OUTDIR, "fci_logs_pag_legend.txt")
with open(legend_path, "w") as f:
    for i, name in enumerate(var_order, 1):
        f.write(f"X{i} = {name}\n")

print("Wrote:", png_plain)
print("Wrote:", png_sink)
print("Wrote:", png_sink_pruned)
print("Legend:", legend_path)

