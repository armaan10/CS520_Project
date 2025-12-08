# ~/theta_project/scripts_cs520/eda_plots_final.py
import os, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --------- paths ----------
DATA_DIR = os.path.expanduser('/Users/sdarmora/theta_project/data/log_data/filter_data')
OUT_DIR  = os.path.expanduser('~/theta_project/scripts_cs520/eda_plots_final')
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
FILES = {y: f'cleaned_joblog_{y}1231.csv' for y in YEARS}

# --------- helpers ----------
def _safe_seconds_from_timedelta_str(series):
    ts = pd.to_timedelta(series, errors='coerce')
    return ts.dt.total_seconds()

def _ensure_core_cols(df):
    # EXIT_FAIL
    if 'EXIT_FAIL' not in df.columns:
        if 'EXIT_STATUS' in df.columns:
            df['EXIT_FAIL'] = (pd.to_numeric(df['EXIT_STATUS'], errors='coerce').fillna(0) != 0).astype(int)
        else:
            df['EXIT_FAIL'] = 0
    # JOB_DURATION_SEC
    if 'JOB_DURATION_SEC' not in df.columns:
        if 'JOB_DURATION' in df.columns:
            df['JOB_DURATION_SEC'] = _safe_seconds_from_timedelta_str(df['JOB_DURATION'])
        else:
            df['JOB_DURATION_SEC'] = np.nan
    # timestamps
    for c in ['START_TIMESTAMP', 'END_TIMESTAMP']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    # numeric core fields
    for c in ['NODES_REQUESTED','CORES_REQUESTED','NODES_USED','CORES_USED',
              'RUNTIME_SECONDS','WALLTIME_SECONDS','JOB_DURATION_SEC']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def load_year(y):
    path = os.path.join(DATA_DIR, FILES[y])
    df = pd.read_csv(path, low_memory=False)
    df['YEAR'] = y
    return _ensure_core_cols(df)

def load_all():
    parts = [load_year(y) for y in YEARS]
    return pd.concat(parts, ignore_index=True)

def savefig(path, tight=True):
    if tight: plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

# ---------- PLOT 1: jobs/week + failure rate/week (dual axis, distinct colors) ----------
def plot_weekly_counts_and_failure(df):
    if 'START_TIMESTAMP' not in df.columns:
        return
    d = df[['START_TIMESTAMP','EXIT_FAIL']].dropna().copy()
    d = d.set_index('START_TIMESTAMP').sort_index()
    weekly = d.resample('W-MON').agg(jobs=('EXIT_FAIL','size'), fail=('EXIT_FAIL','mean')).reset_index()

    fig, ax = plt.subplots(figsize=(12,4))
    ln1, = ax.plot(weekly['START_TIMESTAMP'], weekly['jobs'], label='Jobs/week (left)', color='tab:blue', linewidth=1.5)
    ax.set_ylabel('Jobs per week')
    ax.set_xlabel('Week')

    ax2 = ax.twinx()
    ln2, = ax2.plot(weekly['START_TIMESTAMP'], weekly['fail'], label='Failure rate (right)', color='tab:red', linewidth=1.5)
    ax2.set_ylabel('Failure rate')
    ax2.set_ylim(0, 1)

    lines = [ln1, ln2]
    ax.legend(lines, [l.get_label() for l in lines], loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    savefig(os.path.join(OUT_DIR, 'counts_failure_by_week.png'))

# ---------- PLOT 2: pooled histograms of log10(time) with unified axes ----------
def plot_pooled_log_hists(df):
    cols = [('WALLTIME_SECONDS','Log₁₀(Walltime [s])'),
            ('RUNTIME_SECONDS','Log₁₀(Runtime [s])'),
            ('JOB_DURATION_SEC','Log₁₀(Duration [s])')]
    logvals = []
    for c,_ in cols:
        if c in df.columns:
            v = df[c].dropna()
            v = v[v>0]
            logvals.append(np.log10(v))
        else:
            logvals.append(pd.Series(dtype=float))

    xmin = 0
    xmax = max([v.max() if not v.empty else 0 for v in logvals] + [6])

    fig, axes = plt.subplots(1,3, figsize=(12,3.5), sharey=True)
    for ax, (c, title), vals in zip(axes, cols, logvals):
        ax.hist(vals, bins=100, alpha=0.9)
        ax.set_xlim(xmin, xmax)
        ax.set_title(title)
        ax.set_ylabel('Count')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        if not vals.empty:
            med = np.median(vals)
            q1, q3 = np.percentile(vals, [25,75])
            ax.axvline(med, linestyle='--', linewidth=1)
            ax.axvspan(q1, q3, alpha=0.1)
    savefig(os.path.join(OUT_DIR, 'log_time_hists_pooled.png'))

# ---------- PLOT 3/4: request → used (cores, nodes) per year ----------
def _per_year_scatter(df, x, y, title_prefix, fname):
    fig, axes = plt.subplots(3,3, figsize=(12,10))
    axes = axes.ravel()
    for i, yyear in enumerate(YEARS):
        ax = axes[i]
        sub = df[df['YEAR']==yyear]
        sub = sub[[x,y]].dropna()
        ax.scatter(sub[x], sub[y], s=6, alpha=0.6)
        mn = min(sub[x].min(), sub[y].min()) if not sub.empty else 0
        mx = max(sub[x].max(), sub[y].max()) if not sub.empty else 1
        ax.plot([mn,mx],[mn,mx], linewidth=1)
        ax.set_title(f'{title_prefix} — {yyear}')
        ax.set_xlabel(x); ax.set_ylabel(y)
    for j in range(len(YEARS), len(axes)):
        axes[j].axis('off')
    savefig(os.path.join(OUT_DIR, fname))

def plot_request_used_panels(df):
    _per_year_scatter(df, 'CORES_REQUESTED','CORES_USED', 'Cores requested→used', 'cores_req_used_by_year.png')
    _per_year_scatter(df, 'NODES_REQUESTED','NODES_USED', 'Nodes requested→used', 'nodes_req_used_by_year.png')

# ---------- PLOT 5: log(walltime) vs log(runtime) per year ----------
def plot_log_wall_vs_runtime(df):
    if not {'WALLTIME_SECONDS','RUNTIME_SECONDS'}.issubset(df.columns): return
    fig, axes = plt.subplots(3,3, figsize=(12,10))
    axes = axes.ravel()
    for i,yyear in enumerate(YEARS):
        ax = axes[i]
        sub = df[df['YEAR']==yyear][['WALLTIME_SECONDS','RUNTIME_SECONDS']].dropna()
        sub = sub[(sub>0).all(axis=1)]
        if sub.empty:
            ax.axis('off'); continue
        x = np.log10(sub['WALLTIME_SECONDS'])
        y = np.log10(sub['RUNTIME_SECONDS'])
        hb = ax.hexbin(x, y, gridsize=50, mincnt=1)
        ax.set_title(f'log(walltime)→log(runtime) — {yyear}')
        ax.set_xlabel('log(walltime)'); ax.set_ylabel('log(runtime)')
    for j in range(len(YEARS), len(axes)):
        axes[j].axis('off')
    savefig(os.path.join(OUT_DIR, 'logwall_logrun_by_year_hex.png'))

# ---------- PLOT 6/7/8: correlation heatmaps (Pearson, Spearman, |S|-|P|) ----------
def plot_correlations(df):
    cols = ['NODES_REQUESTED','CORES_REQUESTED','NODES_USED','CORES_USED',
            'RUNTIME_SECONDS','WALLTIME_SECONDS','JOB_DURATION_SEC']
    avail = [c for c in cols if c in df.columns]
    M = df[avail].apply(pd.to_numeric, errors='coerce').dropna(how='all')
    M = M.fillna(0)

    pear = M.corr(method='pearson')
    spear = M.corr(method='spearman')
    diff = spear.abs() - pear.abs()

    def show_heat(mat, title, fname, vmin=0, vmax=1):
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(mat.values, vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_xticks(range(len(mat.columns))); ax.set_yticks(range(len(mat.index)))
        ax.set_xticklabels(mat.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(mat.index, fontsize=9)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        savefig(os.path.join(OUT_DIR, fname))
    show_heat(pear, 'Pearson (pooled)', 'corr_pearson.png')
    show_heat(spear,'Spearman (pooled)', 'corr_spearman.png')
    show_heat(diff, '|Spearman|-|Pearson| (pooled)', 'corr_spearman_minus_pearson_abs.png', vmin=-0.2, vmax=0.2)

# ---------- PLOT 9: yearly medians + failure rate with fixed colors & legend ----------
def plot_yearly_medians_and_failure(df):
    g = df.groupby('YEAR')
    med_log_run  = np.log10(g['RUNTIME_SECONDS'].median().clip(lower=1)).reindex(YEARS)
    med_log_wall = np.log10(g['WALLTIME_SECONDS'].median().clip(lower=1)).reindex(YEARS)
    med_nodes    = g['NODES_USED'].median().reindex(YEARS)
    fail_rate    = g['EXIT_FAIL'].mean().reindex(YEARS)

    fig, ax = plt.subplots(figsize=(12,4))
    l1, = ax.plot(YEARS, med_log_run,  label='median log(runtime)',  color='tab:blue',  marker='o')
    l2, = ax.plot(YEARS, med_log_wall, label='median log(walltime)', color='tab:orange', marker='o')
    l3, = ax.plot(YEARS, med_nodes,    label='median nodes_used',    color='tab:green', marker='o')
    ax.set_xlabel('Year'); ax.set_ylabel('Median value'); ax.grid(axis='y', linestyle='--', alpha=0.3)

    ax2 = ax.twinx()
    l4, = ax2.plot(YEARS, fail_rate, label='failure rate', color='tab:red', marker='o')
    ax2.set_ylabel('Failure rate'); ax2.set_ylim(0, 1)

    lines = [l1,l2,l3,l4]
    ax.legend(lines, [ln.get_label() for ln in lines], loc='upper right')
    savefig(os.path.join(OUT_DIR, 'yearly_medians_with_failure.png'))

# ---------- main ----------
if __name__ == '__main__':
    df_all = load_all()

    # 1) weekly jobs + failure (fixed colors/legend)
    plot_weekly_counts_and_failure(df_all)

    # 2) pooled log-histograms with unified axes + med/IQR markers
    plot_pooled_log_hists(df_all)

    # 3) cores requested→used per year
    plot_request_used_panels(df_all)

    # 4) walltime vs runtime per year (hexbin density)
    plot_log_wall_vs_runtime(df_all)

    # 5) correlation heatmaps
    plot_correlations(df_all)

    # 6) yearly medians + failure rate (fixed colors/legend)
    plot_yearly_medians_and_failure(df_all)

    print(f'Wrote figures to: {OUT_DIR}')

