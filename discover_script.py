import pandas as pd 
import numpy as np
import sys
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

from causallearn.utils.cit import gsq
from causallearn.search.ScoreBased.GES import ges
from matplotlib import pyplot as plt
from causallearn.utils.GraphUtils import GraphUtils
from sklearn.preprocessing import StandardScaler
from causallearn.score.LocalScoreFunction import local_score_BDeu
from causallearn.score.LocalScoreFunction import local_score_BIC


#RUN FCI GES PC
#All discrete, all continous, mixed 
#kci for non-linear mixed data
#g2 for discrete
#ges Local bic score for mixed data 

def discrete_only(df,algo='FCI',a = 0.01):
    select_cols = ['nodes_alloc', 
    'cpus_alloc', 
    'mem_alloc', 
    'partition',
    'num_alloc_gpus',
    'runtime',
    'system_load',
    'node_hours',
    'user_job_frequency',
    'status'] 
    df_res = df[select_cols]
    df_res['partition'] = df_res['partition'].astype('category')
    df_res['partition'] = df_res['partition'].cat.codes
    df_res['status'] = df_res['status'].astype('category').cat.codes
    df_res['runtime_bin'] = pd.cut(np.log1p(df_res['runtime']), bins=5, labels=False)
    df_res['mem_alloc_bin'] = pd.qcut(np.log1p(df_res['mem_alloc']), q=4, labels=False)
    df_res['node_hours_bin'] = pd.cut(np.log1p(df_res['node_hours']), bins=5, labels=False)
    final_df = df_res[[
    'nodes_alloc', 'cpus_alloc', 'mem_alloc_bin',
    'num_alloc_gpus', 
    'runtime_bin', 
    'node_hours_bin',
     'status', 
     'user_job_frequency',
     'system_load'
    ]]
    data = final_df.to_numpy()
    if algo.upper() == 'FCI':
        print("Running FCI")

        g,edges = fci(data, gsq, alpha= a)
        output_file = f'result_fci_discrete_{a}.jpg'
        pdy = GraphUtils.to_pydot(g,labels=final_df.columns)
        pdy.write_jpg(output_file)
    elif algo.upper() == 'PC':
        print("Running PC")
        cg = pc(data,indep_test= gsq, alpha = a)
        
        pyd = GraphUtils.to_pydot(cg.G,labels=final_df.columns)
        output_file = f'result_pc_discrete_{a}.png'
        pyd.write_png(output_file)
    elif algo.upper() == 'GES':
        print("Running GES")
        
        record = ges(data,score_func='local_score_BDeu')
        pyd = GraphUtils.to_pydot(record['G'], labels = final_df.columns)
        pyd.write_png('result_ges_discrete.png')
    else:
        print("No match check args")
def mixed_data (df,algo = 'FCI' , a = 0.05):
    cols_to_norm = ['nodes_alloc', 
    'cpus_alloc', 
    'mem_alloc', 
    'num_alloc_gpus',
    'runtime',
    'system_load',
    'node_hours',
    'user_job_frequency'] 
    select_cols = ['nodes_alloc', 
        'cpus_alloc', 
        'mem_alloc', 
        'num_alloc_gpus',
        'runtime',
        'system_load',
        'node_hours',
        'user_job_frequency',
        'status'] 
    df_res = df[select_cols]
    del df
    scaler = StandardScaler()
    df_res[cols_to_norm] = scaler.fit_transform(df_res[cols_to_norm]) 
    data = df_res.to_numpy()


    if algo.upper() == 'FCI':
        print('RUnning FCI')
        g,edges = fci(data, fisherz, alpha= alpha)
        output_file = f'result_fci_cont_{a}.jpg'
        pdy = GraphUtils.to_pydot(g,labels=df_res.columns)
        pdy.write_jpg(output_file)
    elif algo.upper() == 'PC':
        print('RUnning PC')

        cg = pc(data,indep_test= fisherz, alpha = alpha)
        pyd = GraphUtils.to_pydot(cg.G,labels=df_res.columns)
        output_file = f'result_pc_cont_{a}.png'
        pyd.write_png(output_file)
    elif algo.upper() == 'GES':
        print('RUnning GES')

        record = ges(data)

        pyd = GraphUtils.to_pydot(record['G'], labels = df_res.columns)
        pyd.write_png('result_ges_cont.png')
if __name__ == "__main__" :
    type_of_method=sys.argv[1]
    algo = sys.argv[2]
    alpha = float(sys.argv[3])
    path = "/home/cc/CS520_Project/slurm_log_buff.csv"
    df = pd.read_csv(path)
    print(type(alpha))
    if type_of_method =="dis":
        discrete_only(df,algo,alpha)
    else:
        mixed_data(df,algo, alpha)