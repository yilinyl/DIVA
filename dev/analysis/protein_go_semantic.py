# Generic imports
import os, sys
from pathlib import Path
import re
import signal
import pickle
import json
import fire
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

from collections import defaultdict, Counter
from goatools.obo_parser import GODag
from goatools.semantic import TermCounts

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('..'))

from analysis.semantic_utils import agg_lin_similarity, run_with_timeout, timeout_handler


def pairwise_similarity(prot_list, prot_anno_dict, go_dag, term_counts, signum=2):
    """
    Pairwise semantic similarity among all proteins in input list according to associated GO terms
    """

    # Set the handler for the alarm signal
    
    signal.signal(signal.SIGALRM, timeout_handler)
    sim_lst = []
    for i, prot1 in enumerate(prot_list):
        print('Processing', prot1)
        for prot2 in prot_list[i+1: ]:
            try:
                lin_sim_cur = run_with_timeout(agg_lin_similarity, signum=signum, term_set1=prot_anno_dict[prot1], term_set2=prot_anno_dict[prot2], onto_dag=go_dag, termcounts=term_counts)
                sim_lst.append(lin_sim_cur)
            except TimeoutError:
                print('Timeout for {} {}'.format(prot1, prot2))
                sim_lst.append(-1)
                continue
            # sim_lst.append(agg_lin_similarity(prot_anno_dict[prot1], prot_anno_dict[prot2], go_dag, term_counts))

    return squareform(np.array(sim_lst), checks=False) + np.eye(len(prot_list))

    
def main(var_data_dir='/home/yl986/data/variant_data/',
         dataset_dir='pred/disease_clean1',
         prot_summary_file='protein_stats.txt',
         ontology_obo='disease_ontology/GO/go-basic.obo',
         anno_file='disease_ontology/parsed/prot_GO_CC.pkl'):
    
    var_data_root = Path(var_data_dir)
    dataset_root = var_data_root / dataset_dir
    prot_stats = pd.read_csv(dataset_root / prot_summary_file, sep='\t')

    go_dag = GODag(var_data_root / ontology_obo)
    with open(var_data_root / anno_file, 'rb') as f:
        prot_anno_dict = pickle.load(f)

    prot_selected = prot_stats[prot_stats['UniProt'].isin(prot_anno_dict)].query('n_dis_var > 0')['UniProt'].tolist()
    print('Computing semantic similarity over {} proteins'.format(len(prot_selected)))
    termcounts = TermCounts(go_dag, prot_anno_dict)
    lin_sim_mat = pairwise_similarity(prot_selected, prot_anno_dict, go_dag, termcounts, signum=2)
    split = os.path.basename(anno_file).split('.')[0][-2:].lower()

    with open(dataset_root / f'semantic_sim/dis_prot_by_{split}1.pkl', 'wb') as f:
        pickle.dump({'prot': prot_selected, 'lin_sim_mat': lin_sim_mat}, f)


if __name__ == '__main__':
    # var_data_root = Path('/home/yl986/data/variant_data/')
    # dataset_root = var_data_root / 'pred/disease_clean1/'
    fire.Fire(main)
    