# theta_eda_all_years.py

import os, math, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILES = [
    "/Users/sdarmora/theta_project/data/log_data/filter_data/cleaned_joblog_20171231.csv",
    "/Users/sdarmora/theta_project/data/log_data/filter_data/cleaned_joblog_20181231.csv",
    "/Users/sdarmora/theta_project/data/log_data/filter_data/cleaned_joblog_20191231.csv",
    "/Users/sdarmora/theta_project/data/log_data/filter_data/cleaned_joblog_20201231.csv",
    "/Users/sdarmora/theta_project/data/log_data/filter_data/cleaned_joblog_20211231.csv",
    "/Users/sdarmora/theta_project/data/log_data/filter_data/cleaned_joblog_20221231.csv",
    "/Users/sdarmora/theta_project/data/log_data/filter_data/cleaned_joblog_20231231.csv",
]

OUTDIR = "/Users/sdarmora/theta_project/results/eda_plots"
os.makedirs(OUTDIR, exist_ok=True)

COL_RUNTIME = "RUNTIME_SECONDS"
COL_WALL    = "WALLTIME_SECONDS"
COL_DURSEC  = "JOB_DURATION_SEC"
COL_DURSTR  = "JOB_DURATION"
COL_NODES_R = "NODES_REQUESTED"
COL_CORES_R = "CORES_REQUESTED"
COL_NODES_U = "NODES_USED"
COL_CORES_U = "CORES_USED"
COL_EXIT    = "EXIT_STATUS"
COL_QUEUE   = "QUEUE_NAME"
COL_START   = "START_TIMESTAMP"
COL_END     = "END_TIMESTAMP"

BASE_NUMS = [COL_NODES_R, COL_CORES_R, COL_NODES_U, COL_CORES_U, COL_RUNTIME, COL_WALL]

def read_one(path):
    df = pd.read_csv(path, low_memory=False)
    if COL_DURSEC not in df.columns and COL_DURSTR in df.columns:
        td = pd.to_timedelta(df[COL_DURSTR], errors="coerce")
        df[COL_DURSEC] = td.dt.total_seconds()
    for c in [COL_RUNTIME, COL_WALL, COL_DURSEC, COL_NODES_R, COL_CORES_R, COL_NODES_U, COL_CORES_U]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if COL_EXIT in df.columns:
        df["EXIT_FAIL"] = (pd.to_numeric(df[COL_EXIT], errors="coerce").fillna(0) != 0).astype(int)
    else:
        df["EXIT_FAIL"] = np.nan
    for tcol in [COL_START, COL_END]:
        if tcol in df.columns:
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    year = None
    base = os.path.basename(path)
    for y in range(2017, 2030):
        if str(y) in base:
            year = y
            break
    if year is None:
        if COL_START in df.columns and df[COL_START].notna().any():
            year = int(df[COL_START].dt.year.mode().iloc[0])
        elif COL_END in df.columns and df[COL_END].notna().any():
            year = int(df[COL_END].dt.year.mode().iloc[0])
    df["YEAR_TAG"] = year
    return df

dfs = [read_one(p) for p in INPUT_FILES]
pooled = pd.concat(dfs, ignore_index=True)

def log1p(x):
    x = pd.to_numeric(x, errors="coerce")
    return np.log10(x.clip(lower=0) + 1.0)

pooled["log_runtime"]   = log1p(pooled[COL_RUNTIME]) if COL_RUNTIME in pooled.columns else np.nan
pooled["log_walltime"]  = log1p(pooled[COL_WALL])    if COL_WALL in pooled.columns else np.nan
if COL_DURSEC in pooled.columns:
    pooled["log_duration"] = log1p(pooled[COL_DURSEC])

# -------- Page 1.A: job counts + fail rate over time --------
ts_col = COL_START if COL_START in pooled.columns else (COL_END if COL_END in pooled.columns else None)
if ts_col is not None:
    ts = pooled[[ts_col, "EXIT_FAIL"]].dropna(subset=[ts_col]).copy()
    ts["WEEK"] = ts[ts_col].dt.to_period("W").dt.start_time
    grp = ts.groupby("WEEK", as_index=False).agg(
        jobs=("EXIT_FAIL","size"),
        fail=("EXIT_FAIL","mean")
    )
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(grp["WEEK"], grp["jobs"])
    ax1.set_ylabel("Jobs per week")
    ax1.set_xlabel("Week")
    ax2 = ax1.twinx()
    ax2.plot(grp["WEEK"], grp["fail"])
    ax2.set_ylabel("Failure rate")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "counts_failrate_by_week.png"), dpi=200)
    plt.close(fig)

# -------- Page 1.B: pooled distributions (log) --------
cols_for_dist = []
if "log_walltime" in pooled.columns: cols_for_dist.append("log_walltime")
if "log_runtime" in pooled.columns:  cols_for_dist.append("log_runtime")
if "log_duration" in pooled.columns: cols_for_dist.append("log_duration")
if cols_for_dist:
    fig, axes = plt.subplots(1, len(cols_for_dist), figsize=(4*len(cols_for_dist), 3))
    if len(cols_for_dist)==1: axes=[axes]
    for ax, c in zip(axes, cols_for_dist):
        s = pooled[c].dropna()
        ax.hist(s, bins=60)
        ax.set_title(c)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "dist_pooled_log_walltime_runtime_duration.png"), dpi=200)
    plt.close(fig)

# -------- Page 2: Pearson / Spearman / Nonlinearity (pooled) --------
heat_vars = BASE_NUMS.copy()
if COL_DURSEC in pooled.columns:
    heat_vars.append(COL_DURSEC)
dfh = pooled[heat_vars].apply(pd.to_numeric, errors="coerce")
pear = dfh.corr(method="pearson", min_periods=100)
spear = dfh.corr(method="spearman", min_periods=100)
nl = spear.abs() - pear.abs()

def plot_heat(mat, title, fname):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mat.index)));   ax.set_yticklabels(mat.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, fname), dpi=220)
    plt.close(fig)

plot_heat(pear,  "Pearson (pooled)",   "heatmap_pearson_pooled.png")
plot_heat(spear, "Spearman (pooled)",  "heatmap_spearman_pooled.png")
plot_heat(nl,    "|Spearman|-|Pearson| (pooled)", "heatmap_nonlinearity_pooled.png")

# -------- Page 3.A: yearly medians (runtime, walltime, nodes_used, fail rate) --------
year_ok = pooled["YEAR_TAG"].notna()
if year_ok.any():
    Y = pooled.loc[year_ok, :]
    med_tbl = []
    for y, g in Y.groupby("YEAR_TAG"):
        row = {"YEAR": int(y)}
        if COL_RUNTIME in g.columns: row["median_runtime_log"] = g["log_runtime"].median(skipna=True)
        if COL_WALL    in g.columns: row["median_walltime_log"] = g["log_walltime"].median(skipna=True)
        if COL_NODES_U in g.columns: row["median_nodes_used"]   = g[COL_NODES_U].median(skipna=True)
        if "EXIT_FAIL" in g.columns: row["fail_rate"]           = g["EXIT_FAIL"].mean(skipna=True)
        med_tbl.append(row)
    med = pd.DataFrame(med_tbl).sort_values("YEAR")
    fig, ax = plt.subplots(figsize=(10,4))
    if "median_runtime_log" in med.columns: ax.plot(med["YEAR"], med["median_runtime_log"], label="median log(runtime)")
    if "median_walltime_log" in med.columns: ax.plot(med["YEAR"], med["median_walltime_log"], label="median log(walltime)")
    if "median_nodes_used" in med.columns: ax.plot(med["YEAR"], med["median_nodes_used"], label="median nodes_used")
    ax.set_xlabel("Year"); ax.set_ylabel("Median value")
    ax.legend(loc="best")
    ax2 = ax.twinx()
    if "fail_rate" in med.columns: ax2.plot(med["YEAR"], med["fail_rate"])
    ax2.set_ylabel("Failure rate")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "yearly_medians_runtime_walltime_nodesused_failrate.png"), dpi=200)
    plt.close(fig)

# -------- Page 3.B: Request→Use funnels by year (hexbin) --------
def panel_hex(df, x, y, fname, title):
    groups = list(df.groupby("YEAR_TAG"))
    if not groups: return
    n = len(groups)
    cols = min(3, n)
    rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows), squeeze=False)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]; ax.axis("on")
            if idx < n:
                yv, g = groups[idx]
                gx = pd.to_numeric(g[x], errors="coerce")
                gy = pd.to_numeric(g[y], errors="coerce")
                good = gx.notna() & gy.notna()
                if good.any():
                    ax.hexbin(gx[good], gy[good], gridsize=35, mincnt=1)
                ax.set_title(f"{title} — {int(yv)}")
                ax.set_xlabel(x); ax.set_ylabel(y)
            else:
                ax.axis("off")
            idx += 1
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, fname), dpi=200)
    plt.close(fig)

if year_ok.any():
    Y = pooled.loc[year_ok, :]
    if COL_NODES_R in Y.columns and COL_NODES_U in Y.columns:
        panel_hex(Y, COL_NODES_R, COL_NODES_U, "funnel_nodes_req_used_by_year.png", "Nodes requested→used")
    if COL_CORES_R in Y.columns and COL_CORES_U in Y.columns:
        panel_hex(Y, COL_CORES_R, COL_CORES_U, "funnel_cores_req_used_by_year.png", "Cores requested→used")
    if "log_walltime" in Y.columns and "log_runtime" in Y.columns:
        tmp = Y[["YEAR_TAG","log_walltime","log_runtime"]].dropna()
        groups = list(tmp.groupby("YEAR_TAG"))
        n = len(groups)
        cols = min(3, n)
        rows = math.ceil(n/cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows), squeeze=False)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                ax = axes[r][c]
                if idx < n:
                    yv, g = groups[idx]
                    ax.hexbin(g["log_walltime"], g["log_runtime"], gridsize=35, mincnt=1)
                    ax.set_title(f"log(walltime)→log(runtime) — {int(yv)}")
                    ax.set_xlabel("log(walltime)")
                    ax.set_ylabel("log(runtime)")
                else:
                    ax.axis("off")
                idx += 1
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, "funnel_walltime_runtime_by_year.png"), dpi=200)
        plt.close(fig)

# -------- Page 4.A: fail rate by walltime decile (pooled) --------
if "log_walltime" in pooled.columns and "EXIT_FAIL" in pooled.columns:
    t = pooled[["log_walltime","EXIT_FAIL"]].dropna()
    if len(t) > 0:
        t["decile"] = pd.qcut(t["log_walltime"], q=10, labels=False, duplicates="drop")
        g = t.groupby("decile", as_index=False)["EXIT_FAIL"].mean()
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(g["decile"].astype(int), g["EXIT_FAIL"])
        ax.set_xlabel("log(walltime) decile")
        ax.set_ylabel("Failure rate")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, "failrate_by_walltime_decile_pooled.png"), dpi=200)
        plt.close(fig)

# -------- Page 4.B: fail rate by queue (pooled, if available) --------
if COL_QUEUE in pooled.columns and "EXIT_FAIL" in pooled.columns:
    q = pooled[[COL_QUEUE,"EXIT_FAIL"]].dropna()
    if len(q) > 0:
        g = q.groupby(COL_QUEUE, as_index=False)["EXIT_FAIL"].mean().sort_values("EXIT_FAIL", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.barh(g[COL_QUEUE], g["EXIT_FAIL"])
        ax.invert_yaxis()
        ax.set_xlabel("Failure rate")
        ax.set_ylabel("QUEUE_NAME")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, "failrate_by_queue_pooled.png"), dpi=220)
        plt.close(fig)

print("EDA plots written to:", OUTDIR)

