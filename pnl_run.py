#pnl
from causallearn.search.FCMBased.PNL.PNL import PNL
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
pc_df = pd.read_csv('mixed_pillai_.csv')
pnl = PNL()
pairs = [['runtime', 'system_load'],['nodes_alloc','cpus_alloc'],['nodes_alloc','mem_alloc'],['cpus_alloc','mem_alloc']]


def run_pnl(pair):
    print(f"Processing pair: {pair}")
    col_x, col_y = pair
    data_x = pc_df[col_x].to_numpy().reshape(-1, 1)
    data_y = pc_df[col_y].to_numpy().reshape(-1, 1)

    pnl = PNL()                      # IMPORTANT: create a new PNL per thread
    p_fwd, p_bwd = pnl.cause_or_effect(data_x, data_y)

    direction = f"{col_x} -> {col_y}" if p_fwd > p_bwd else f"{col_y} -> {col_x}"
    return pair, p_fwd, p_bwd, direction


# ---- Run with 4 threads ----
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_pnl, pairs))


# ---- Print results ----
for pair, p_fwd, p_bwd, direction in results:
    print(f"Pair: {pair}, P-Value Forward: {p_fwd}, P-Value Backward: {p_bwd}")
    print(f"Inferred Causal Direction: {direction}")
    print()