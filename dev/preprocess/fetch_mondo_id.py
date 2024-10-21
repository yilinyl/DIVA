# Generic imports
import os, sys
from pathlib import Path
import json

import pandas as pd

from collections import defaultdict
from ontobio.ontol_factory import OntologyFactory

if __name__ == '__main__':
    var_data_root = Path('/home/yl986/data/variant_data/')
    df_var = pd.read_csv(var_data_root / 'disvar_comb4.csv')
    var_dis_vocab = df_var[['phenotype']].drop_duplicates().dropna().reset_index(drop=True)

    # Load PrimeKG disease data
    primekg_raw = pd.read_csv(var_data_root / 'PrimeKG/disease_features.csv')
    # fill mondo_definition > umls_definition > orphanet_definition
    primekg_raw["definition"] = primekg_raw["mondo_definition"].fillna(primekg_raw["umls_description"])
    primekg_raw["definition"] = primekg_raw["definition"].fillna(primekg_raw["orphanet_definition"])
    primekg_clean = primekg_raw[['node_index', 'mondo_id', 'mondo_name', 'definition', 'mondo_definition']].drop_duplicates().reset_index(drop=True)

    # Retrieve information from Mondo Ontology
    # Load the MONDO ontology
    print("Build Ontology...")
    ont_factory = OntologyFactory()
    mondo = ont_factory.create(f'{str(var_data_root)}/disease_ontology/mondo.obo')

    mondo_dis_vocab = pd.read_csv(var_data_root / 'disease_ontology/primekg_disease_vocab_extend.csv')  # load parsed disease terms
    vocab_remain = var_dis_vocab[~(var_dis_vocab['phenotype'].isin(mondo_dis_vocab['name_lower']) | var_dis_vocab['phenotype'].isin(primekg_clean['mondo_name'].str.lower()))]['phenotype'].tolist()

    remain_dis2id = dict()
    catch_list = []
    for i, term in enumerate(vocab_remain):
        if (i+1) % 100 == 0:
            print(f"{i+1} / {len(vocab_remain)} processed!")
        matches = mondo.search(term, is_partial_match=True, synonyms=True)
        if not matches:
            catch_list.append(term)
        remain_dis2id[term] = matches

    with open(var_data_root / 'disease_ontology/remain_disease_id_dict.json', 'w') as f:
        json.dump(remain_dis2id, f, indent=2)
    
    print('Finish retrieving disease ID: {} mapped to Mondo ID, {} missed'.format(len(vocab_remain) - len(catch_list), len(catch_list)))