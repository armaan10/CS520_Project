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
from pgmpy.estimators import PC
from pgmpy.estimators import GES
from pgmpy.estimators import ExpertKnowledge
import networkx as nx   
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
    #'user_job_frequency',
    'status'] 
    df_res = df[select_cols]
    labels_nodes = ["single","small","medium","large"]
    nodes_alloc_bins = [0,1,10,100,np.inf]
    label = ['small','medium','large']
    cpus_alloc_bins = [0, 4, 16,np.inf]
    mem_alloc_bin =[0,50*1024,1*1024*1024,np.inf]

    conditions = [
    df_res["num_alloc_gpus"] == 0,
    df_res["num_alloc_gpus"] == 1,
    df_res["num_alloc_gpus"] > 1
    ]

    choices = ["cpu_only", "single_gpu", "multi_gpu"]
    df_res['gpu_cat'] = np.select(conditions, choices, default="unknown")
    df_res['gpu_cat'] = df_res['gpu_cat'].astype('category')
    df_res['nodes_alloc_cat'] = pd.cut(df_res['nodes_alloc'], bins=nodes_alloc_bins, labels=labels_nodes).astype('category')
    df_res['cpus_alloc_cat'] = pd.cut(df_res['cpus_alloc'], bins=cpus_alloc_bins, labels=label).astype('category')
    df_res['mem_alloc_cat'] = pd.cut(df_res['mem_alloc'], bins=mem_alloc_bin, labels=label).astype('category')
    df_res['partition'] = df_res['partition'].astype('category')
    #df_res['partition'] = df_res['partition'].cat.codes
    df_res['status'] = df_res['status'].astype('category')
    df_res['runtime_bin'] = pd.cut(np.log1p(df_res['runtime']), bins=3, labels=label)
    #df_res['mem_alloc_bin'] = pd.qcut(np.log1p(df_res['mem_alloc']), q=4, labels=False)
    df_res['node_hours_bin'] = pd.cut(np.log1p(df_res['node_hours']), bins=3, labels=label)
    df_res['system_load_cat'] = pd.cut(df_res['system_load'], bins=3, labels=label).astype('category')
    final_df = df_res[[
    'nodes_alloc_cat',
     'cpus_alloc_cat', 
     
    'gpu_cat', 
    'mem_alloc_cat',
    #'partition',
    'runtime_bin', 
    'node_hours_bin',
     'status', 
     #'user_job_frequency',
     'system_load_cat'
    ]]
    data = final_df.to_numpy()
    forbid_edges = []
    for i in final_df.columns:
        if i != 'status':
            forbid_edges.append( ('status',i) )
    expert_knowledge = ExpertKnowledge(forbidden_edges=forbid_edges)
    if algo.upper()== 'PC_MIXED':
        print("Running PC Mixed")
        estimated_pdag = PC(final_df).estimate(ci_test='pillai',variant= 'stable')
        estimated_pdag.to_graphviz().draw('pc_disc_mixed_pillai.png', prog='dot',expert_knowledge=expert_knowledge)
    if algo.upper()== 'GES_MIXED':
        print("Running GES Mixed")
        estimated_pdag = GES(final_df).estimate(scoring_method='bic-d',expert_knowledge=expert_knowledge)
        estimated_pdag.to_graphviz().draw('ges_disc_mixed_bic_d_2.png', prog='dot')
        #model= estimated_pdag.to_networkx_graph()
   
        print("GRAPH:", estimated_pdag.graph)
        print("NODES:", list(estimated_pdag.nodes(data=True)))
        print("EDGES:", list(estimated_pdag.edges(data=True)))
        g_clean = nx.DiGraph()
        for n in estimated_pdag.nodes():
            g_clean.add_node(n)
        for u,v in estimated_pdag.edges():  
            g_clean.add_edge(str(u),str(v))
        
        nx.write_gml(g_clean,'ges_disc_mixed_bic_d_2.gml')
        final_df.to_csv('discrete_data_processed_2.csv',index=False)
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
    cols_to_norm = [
    #'nodes_alloc', 
    #'cpus_alloc', 
    #'mem_alloc', 
    #'num_alloc_gpus',
    #'runtime',
    'system_load',
    #'node_hours',
    #'user_job_frequency'
    ] 
    select_cols = ['nodes_alloc', 
        'cpus_alloc', 
        'mem_alloc', 
        'num_alloc_gpus',
        'runtime',
        'system_load',
        'node_hours',
        #'user_job_frequency',
        #'id_user',
        'status'] 
    df['id_user'] = df['id_user'].astype('category')
    df_res = df[select_cols]
    del df
    scaler = StandardScaler()
    df_res[cols_to_norm] = scaler.fit_transform(df_res[cols_to_norm]) 
    data = df_res.to_numpy()
    forbid_edges = []
    for i in df_res.columns:
        if i != 'status':
            forbid_edges.append( ('status',i) )
    expert_knowledge = ExpertKnowledge(forbidden_edges=forbid_edges)
    if algo.upper() == 'PC_MIXED':
        print('RUnning PC Mixed here')
        estimated_pdag = PC(df_res).estimate(ci_test='pillai',variant= 'stable', forbidden_edges=forbid_edges)

        df_res.to_csv('mixed_pillai_.csv',index=False)
        estimated_pdag.to_graphviz().draw('pc_mixed_pillai.png', prog='dot')
        g_clean = nx.DiGraph()
        for n in estimated_pdag.nodes():
            g_clean.add_node(n)
        for u,v in estimated_pdag.edges():  
            g_clean.add_edge(str(u),str(v))
        nx.write_gml(g_clean,'pc_cont_mixed_pillai.gml')

        
    if algo.upper() == 'GES_MIXED':
        print('RUnning GES Mixed')
        estimated_pdag = GES(df_res).estimate(scoring_method='bic-cg')        
        estimated_pdag.to_graphviz().draw('ges_mixed_bic.png', prog='dot')
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
    path = "/home/cc/slurm_log_buff.csv"
    df = pd.read_csv(path)
    print(type(alpha))
    if type_of_method =="dis":
        discrete_only(df,algo,alpha)
    else:
        mixed_data(df,algo, alpha)