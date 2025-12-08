import os, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from causalnex.structure.notears import from_pandas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to cleaned joblog CSV")
    ap.add_argument("--n_sample", type=int, default=30000, help="Max rows to sample for speed")
    ap.add_argument("--w_threshold", type=float, default=0.05, help="Edge weight threshold for pruning")
    ap.add_argument("--outdir", default="~/theta_project/results", help="Output folder")
    args = ap.parse_args()

    csv = os.path.expanduser(args.csv)
    outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # Load data
    df = pd.read_csv(csv)

    # Numeric features
    df["log_runtime"]   = np.log(df["RUNTIME_SECONDS"].clip(lower=1))
    df["log_walltime"]  = np.log(df["WALLTIME_SECONDS"].clip(lower=1))
    start = pd.to_datetime(df["START_TIMESTAMP"])
    end   = pd.to_datetime(df["END_TIMESTAMP"])
    df["JOB_DURATION_SEC"] = (end - start).dt.total_seconds()

    features = [
        "log_runtime","log_walltime",
        "NODES_REQUESTED","CORES_REQUESTED",
        "NODES_USED","CORES_USED",
        "JOB_DURATION_SEC"
    ]
    X = df[features].dropna()

    # Downsample for speed
    if args.n_sample > 0 and len(X) > args.n_sample:
        X = X.sample(n=args.n_sample, random_state=0)

    # Standardize
    Xs = pd.DataFrame(StandardScaler().fit_transform(X.values), columns=features)

    # NOTEARS via CausalNex
    sm = from_pandas(Xs, w_threshold=args.w_threshold)

    # Edges
    edges = list(sm.edges())
    print("Features:", features)
    print(f"Learned {len(edges)} edges:")
    for u, v in edges:
        print(f"{u} -> {v}")

    # Save edges
    out_csv = os.path.join(outdir, f"notears_edges_2018_thr{args.w_threshold}.csv")
    pd.DataFrame(edges, columns=["src","dst"]).to_csv(out_csv, index=False)
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()

