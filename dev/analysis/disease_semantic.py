# Generic imports
import os, sys
from pathlib import Path
import re
import pickle
import json
import fire
import signal
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

from goatools.obo_parser import GODag

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('..'))

from analysis.semantic_utils import agg_lin_similarity, TermCountsHPO, run_with_timeout


# Define a function to handle a single pair of terms
def similarity_single_pair(pair, term_anno_dict, onto_dag, term_counts, signum):
    term1, term2 = pair
    try:
        return run_with_timeout(
            agg_lin_similarity, 
            signum=signum, 
            term_set1=term_anno_dict[term1], 
            term_set2=term_anno_dict[term2], 
            onto_dag=onto_dag, 
            termcounts=term_counts
        )
    except TimeoutError:
        print(f'Timeout for {term1} {term2}')
        return -1


def pairwise_similarity(target_list, term_anno_dict, onto_dag, term_counts, signum=2):
    """
    Pairwise semantic similarity among all proteins in input list according to associated GO terms
    """

    # Set the handler for the alarm signal
    # signal.signal(signal.SIGALRM, timeout_handler)

    sim_lst = []
    with ThreadPoolExecutor() as executor:
        for i, term1 in enumerate(target_list):
            print('Processing', term1)
            futures = [executor.submit(agg_lin_similarity,
                                      term_set1=term_anno_dict[term1],
                                      term_set2=term_anno_dict[term2],
                                      onto_dag=onto_dag,
                                      termcounts=term_counts)
                       for term2 in target_list[i+1:]]
            for future in futures:
                try:
                    sim_lst.append(future.result())
                except TimeoutError:
                    print(f'Timeout for {term1}')
                    sim_lst.append(-1)

    return squareform(np.array(sim_lst), checks=False) + np.eye(len(target_list))

    
def main(var_data_dir='/home/yl986/data/variant_data/',
         dataset_dir='pred/disease_clean1',
         dis_vocab_file='dis_vocab_desc.csv',
         hpo_obo='disease_ontology/HPO/hp.obo',
         anno_file='disease_ontology/parsed/mondo_hpo.pkl'):
    
    var_data_root = Path(var_data_dir)
    dataset_root = var_data_root / dataset_dir
    # prot_stats = pd.read_csv(dataset_root / prot_summary_file, sep='\t')
    df_var_dis = pd.read_csv(dataset_root / dis_vocab_file)

    hpo_dag = GODag(var_data_root / hpo_obo)
    with open(var_data_root / anno_file, 'rb') as f:
        dis_pheno_dict = pickle.load(f)

    mondo_selected = df_var_dis[df_var_dis['mondo_id'].isin(dis_pheno_dict)]['mondo_id'].drop_duplicates().tolist()

    print('Computing semantic similarity over {} diseases'.format(len(mondo_selected)))
    termcounts = TermCountsHPO(hpo_dag, dis_pheno_dict)
    lin_sim_mat = pairwise_similarity(mondo_selected, dis_pheno_dict, hpo_dag, termcounts, signum=5)
    # lin_sim_mat = mp_pairwise_similarity(mondo_selected, dis_pheno_dict, hpo_dag, termcounts, signum=5)

    with open(dataset_root / 'semantic_sim/dis_pheno_sim_1.pkl', 'wb') as f:
        pickle.dump({'mondo_id': mondo_selected, 'lin_sim_mat': lin_sim_mat}, f)


if __name__ == '__main__':
    fire.Fire(main)
    
