import os, sys
from pathlib import Path

import gzip
import pandas as pd

import json
import requests
import time
from tqdm import tqdm


def submit_by_file(fs, rest_url='https://www3.cmbi.umcn.nl/xssp/'):
        
    pdb_file = {'file_': open(fs, 'rb')}
    url_create = '{}api/create/pdb_file/dssp/'.format(rest_url)
    r = requests.post(url_create, files=pdb_file)
    r.raise_for_status()

    job_id = json.loads(r.text)['id']
    
    # struct2job[fname] = job_id
    print(f"Job submitted successfully ID={job_id}")
    return job_id


def submit_by_id(pdb_id, rest_url='https://www3.cmbi.umcn.nl/xssp/'):
        
    url_create = '{}api/create/pdb_id/dssp/'.format(rest_url)
    r = requests.post(url_create, data={'data': pdb_id})
    r.raise_for_status()

    job_id = json.loads(r.text)['id']
    
    # struct2job[fname] = job_id
    print(f"Job submitted successfully ID={job_id}")
    return job_id


def fetch_job_result(job_dict, rest_url='https://www3.cmbi.umcn.nl/xssp/', dssp_out_dir='./'):
    status_dict = {'success': [], 'pending': [], 'fail': []}
    # result_dict = dict()

    dssp_out_root = Path(dssp_out_dir)
    if not dssp_out_root.exists():
        dssp_out_root.mkdir(parents=True)

    for struct_key, job_id in tqdm(job_dict.items()):
        ready = False
        url_status = '{}api/status/pdb_file/dssp/{}/'.format(rest_url, job_id)
        r = requests.get(url_status)
        r.raise_for_status()

        status = json.loads(r.text)['status']
        print("{} Job status: '{}'".format(struct_key, status))

        if status == 'SUCCESS':
            ready = True
            status_dict['success'].append(struct_key)
        elif status in ['FAILURE', 'REVOKED']:
            status_dict['fail'].append(struct_key)
            print(json.loads(r.text)['message'])
            continue
        else:
            status_dict['pending'].append(struct_key)

        if ready:
            url_result = '{}api/result/pdb_file/dssp/{}/'.format(rest_url, job_id)
            r = requests.get(url_result)
            r.raise_for_status()
            result = json.loads(r.text)['result']
            with open(dssp_out_root / f'{struct_key}.dssp', 'w') as f:
                f.write(result)
            # result_dict[struct_key.split('.')[0]] = result
    return status_dict


if __name__ == '__main__':
    var_file = '/local/storage/yl986/3d_vip/data_prepare/data/dataset/full_v1/var_info_full.csv'
    pdb_root_path = Path('/fs/cbsuhyfs1/storage/resources/pdb/data')
    af_root_path = Path('/fs/cbsuhyfs1/storage1/dx38/mutation_pathogenicity/data/AF_structs')
    action = 'fetch'  # one of 'submit' for job submission or 'fetch' for result query
    model_subset = ['PDB']
    overwrite = True
    cov_thres = 0.5
    dssp_out_dir = '/fs/cbsuhyfs1/storage1/yl986/data/DSSP_pdb'
    dssp_out_root = Path(dssp_out_dir)

    if action == 'submit':
        df_var = pd.read_csv(var_file)
        df_var['model'] = df_var['PDB_coverage'].apply(lambda x: 'PDB' if x >= cov_thres else 'AF')
        df_var['struct_id'] = df_var.apply(lambda x: x['UniProt'] if x['model'] == 'AF' else x['PDB'], axis=1)
        if model_subset:
            df_var = df_var[df_var['model'].isin(model_subset)]

        df_var = df_var.drop_duplicates(['struct_id']).reset_index(drop=True)

        struct_job_dict = dict()

        for i, record in tqdm(df_var.iterrows()):
            uprot = record['UniProt']
            model = record['model']
            struct_id = record['struct_id']
            if model == 'PDB':
                struct_key = f'{model}-{struct_id}'
                
            else:
                f_path = af_root_path / 'AF-{}-F1-model_v4.pdb'.format(struct_id)
                
            dssp_f_out = dssp_out_root / f'{struct_key}.dssp'
            if not overwrite and dssp_f_out.exists():
                continue  
            
            if struct_id in struct_job_dict:
                continue
            
            if (i + 1) % 500 == 0:
                time.sleep(5)

            if model == 'PDB':
                try:
                    struct_job_dict[struct_key] = submit_by_id(struct_id)
                except Exception as e:
                    print(e)
            
            else:
                if not f_path.exists():
                    print('Structure file not available for {}-{} (UniProt={})'.format(record['model'], struct_id, uprot))
                    continue
                try:
                    struct_job_dict[struct_key] = submit_by_file(f_path)
                except Exception as e:
                    print(e)
            
        with open('./pdb_dssp_job.json', 'w') as f:
            json.dump(struct_job_dict, f, indent=2)
    
    else:
        with open('./pdb_dssp_job.json', 'r') as f:
            struct_job_dict = json.load(f)
        status = fetch_job_result(struct_job_dict, dssp_out_dir=dssp_out_dir)
        print('Job Summary')
        for key, val in status.items():
            print('{}: {}'.format(key, len(val)))
        
        with open('./status_dict.json', 'w') as f:
            json.dump(status, f, indent=2)


        




