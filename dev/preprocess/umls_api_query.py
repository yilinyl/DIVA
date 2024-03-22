import os, sys
from pathlib import Path

import requests
import time
import json
from tqdm import tqdm
import pandas as pd

API_URL = 'https://uts-ws.nlm.nih.gov/rest/content/current/CUI'
# API_URL = 'https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0009044'
API_KEY = '746fb404-ea8c-4d5b-aba8-c3d253a40b0e'


def check_response(response):
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(response.json())
        raise
        
def query_cui_info(cui):
    # print(uni_id)
    
    query = {'apiKey':API_KEY}

    request = requests.get(url = f"{API_URL}/{cui}", params=query)
    try:
        check_response(request)
    except requests.exceptions.HTTPError as errh:
        print(errh)
    
    results = request.json()
    
    return results
    # results = json.dumps(request.json())
    # try:
    #     results = request.json()['results'][0]
    # except IndexError:
    #     return {'uniprot': uprot, 'gene': na_val, 'name': na_val, 'taxa': na_val, 'description': na_val}, None

def query_def_by_cui(cui):
    # print(uni_id)
    
    query = {'apiKey':API_KEY}

    request = requests.get(url = f"{API_URL}/{cui}/definitions", params=query)
    try:
        check_response(request)
    except requests.exceptions.HTTPError as errh:
        print(errh)
        return
    
    results = request.json()['result']
    cur_dict = dict()

    for res in results:
        if res['rootSource'] == 'MSH':
            return res['value'], res['rootSource']
        elif not res['rootSource'].startswith('MSH'):
            cur_dict[res['rootSource']] = res['value']
    
    for source in ['ORPHANET', 'SNOMEDCT_US', 'HPO', 'NCI']:
        if source in cur_dict:
            return cur_dict[source], source
    
    if cur_dict:
        for k, v in cur_dict.items():
            return v, k
    
        # if res['rootSource'] == 'ORPHANET':
        #     return res['value'], res['rootSource']
        # if res['rootSource'] in ['NCI', 'HPO', 'CSP', 'SNOMEDCT_US']:
        #     return res['value'], res['rootSource']
        # if not res['rootSource'].startswith('MSH'):  # skip MSH definition in other language
        #     return res['value'], res['rootSource']
    
    return results


if __name__ == '__main__':
    limit_num = 150
    sleep_sec = 5
    var_data_root = Path('/home/yl986/data/variant_data')
    # mapped phenotype to CUI
    with open(var_data_root / 'umls/var_pheno_mapped0228.json', 'r') as f:
        pheno2cui = json.load(f)
    cui_list = sorted(set(pheno2cui.values()))

    with open(var_data_root / 'umls/var_cui2name.json', 'r') as f:
        cui2name = json.load(f)
    
    f_out = var_data_root / 'umls/var_cui_desc.txt'
    if f_out.exists():
        df_cache = pd.read_csv(f_out, sep='\t')
        cui_cache = set(df_cache['CUI'])
    else:
        with open(f_out, 'w') as f:
            f.write('\t'.join(['CUI', 'name', 'source', 'definition']) + '\n')
        cui_cache = set()

    with open(f_out, 'a') as f:
        for i, cui in enumerate(cui_list):
            if cui in cui_cache:
                continue

            if (i + 1) % limit_num == 0:
                time.sleep(sleep_sec)
            
            result = query_def_by_cui(cui)
            if result:
                if len(result) != 2:
                    print(f'MSH definition not found for {cui}')
                    continue
                desc, source = result
                cui_cache.add(cui)
                f.write('\t'.join([cui, cui2name[cui], source, desc]) + '\n')
            else:
                print(cui)