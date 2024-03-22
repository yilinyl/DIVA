import os, sys

import urllib
import json
import time
import requests
import pandas as pd


API_URL = "https://rest.uniprot.org"

def check_response(response):
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(response.json())
        raise

def fetch_uniprot_info(uprot, na_val='NA'):
    # print(uni_id)
    request = requests.post(url = f"{API_URL}/uniprotkb/search?query={uprot}")
    try:
        check_response(request)
    except requests.exceptions.HTTPError as errh:
        print(errh)
        
    # results = json.dumps(request.json())
    try:
        results = request.json()['results'][0]
    except IndexError:
        return {'uniprot': uprot, 'gene': na_val, 'name': na_val, 'taxa': na_val, 'description': na_val}, results
    
    prot_name = na_val
    try:
        prot_desc_dict = results['proteinDescription']
        if 'recommendedName' in prot_desc_dict:
            prot_name = prot_desc_dict['recommendedName']['fullName']['value']
        elif 'submissionNames' in prot_desc_dict:
            prot_name = prot_desc_dict['submissionNames'][0]['fullName']['value']
    except KeyError:
        pass
    
    prot_desc = na_val
    try:
        comment_dict = results['comments']
        for com in comment_dict:
            if com['commentType'].upper() == 'FUNCTION':
                prot_desc = com['texts'][0]['value']
                break
    except KeyError:
        pass
    
    gene = na_val
    try:
        gene = results['genes'][0]['geneName']['value']
    except KeyError:
        pass
    
    taxa = na_val
    try:
        taxa = results['organism']['taxonId']
    except KeyError:
        pass
    
    # return {'uniprot': uprot, 'gene': gene, 'name': prot_name, 'taxa': taxa, 'description': prot_desc}, results
    return {'uniprot': uprot, 'gene': gene, 'name': prot_name, 'taxa': taxa, 'description': prot_desc}, results


def fetch_data(uprot_list, limit_num=100, sleep_sec=5):
    prot_info_dict = {}
    for i, uprot in enumerate(uprot_list):
        if (i + 1) % limit_num == 0:
            time.sleep(sleep_sec)
        info_dict, full_res = fetch_uniprot_info(uprot)
        prot_info_dict.update(info_dict)
    
    return prot_info_dict


if __name__ == '__main__':
    f_in = sys.argv[1]
    fname_out = sys.argv[2]
    
    limit_num = 100
    sleep_sec = 5
    update_keys = ['name', 'description']
    start_idx = 9887

    # df_prot = pd.read_csv(f_in, sep='\t')
    with open(f_in, 'r') as f:
        lines = f.read().splitlines()
    
    header = lines.pop(0).split('\t')
    
    cache = set()
    mode = 'w'
    if os.path.exists(fname_out):
        mode = 'a'
        with open(fname_out, 'r') as f:
            cache_lines = f.read().splitlines()
        for l in cache_lines:
            cache.add(l.split('\t', 1)[0])

    with open(fname_out, mode) as f_out:
        if mode == 'w':
            f_out.write('\t'.join(header) + '\n')
        for i, entry in enumerate(lines[start_idx:]):
            cur_dict = dict(zip(header, entry.split('\t')))
            if cur_dict['uniprot'] in cache:
                continue
            if (i + 1) % limit_num == 0:
                time.sleep(sleep_sec)
            info_dict, full_res = fetch_uniprot_info(cur_dict['uniprot'])
            for key in update_keys:
                cur_dict[key] = info_dict[key]
            output = '\t'.join([cur_dict[k] for k in header]) + '\n'
            f_out.write(output)

        
