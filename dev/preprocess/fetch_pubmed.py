# Generic imports
import os, sys
from pathlib import Path
import re
import pickle
import json
import gzip
import time

from Bio import Entrez

MAX_PMID_NUM = 200
SLEEP_INTERVAL = 100
REST_SEC = 5


def fetch_pubmed_ids(query, max_results=10):
    Entrez.email = "your_email@example.com"  # Provide your email address

    # Use ESearch to search for PubMed IDs related to the query
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()

    return record["IdList"]


if __name__ == '__main__':
    data_root = Path('/home/yl986/data/variant_data')
    fname_out = 'pheno2pmid_all.txt'

    with open(data_root / 'umls/var_pheno_mapped0228.json', 'r') as f:
        mapped_pheno_dict = json.load(f)
    
    n_mapped_pheno = len(mapped_pheno_dict)
    with open(data_root / 'medgen/medgen2pmid.pkl', 'rb') as f:
        medgen2pub = pickle.load(f)
    with open(data_root / 'umls/pheno_unmapped.txt', 'r') as f:
        phenos_unmapped = f.read().splitlines()
    cui2mapped_pheno = {v:k for k, v in mapped_pheno_dict.items()}

    cuis_with_pmid = set(medgen2pub.keys()).intersection(cui2mapped_pheno.keys())

    cuis_missing_pmid = list(cui2mapped_pheno.keys() - set(medgen2pub.keys()))
    phenos_missing_pmid = [cui2mapped_pheno[cui] for cui in cuis_missing_pmid]
    print(f'Total disease terms: {n_mapped_pheno + len(phenos_unmapped)}, {n_mapped_pheno} mapped to CUI')
    print(f'{len(phenos_unmapped)} disease term unmapped')
    print(f'Found PubMed information for {n_mapped_pheno - len(cuis_missing_pmid)} (from MedGen), information missing for {len(cuis_missing_pmid)} terms')
    
    pheno2pub_fetched = dict()
    target_phenos = phenos_missing_pmid + phenos_unmapped

    w_mode = 'w'
    if (data_root / fname_out).exists():
        w_mode = 'a'
        with open(data_root / fname_out, 'r') as f:
            cache_lines = f.read().splitlines()
        for l in cache_lines:
            l_split = l.split('\t')
            if len(l_split) > 1:
                pheno2pub_fetched[l_split[0]] = l_split[1:]
            else:
                pheno2pub_fetched[l_split[0]] = []
            
    with open(data_root / fname_out, w_mode) as fout:
        for cid in cuis_with_pmid:
            pmids = medgen2pub[cid]
            if cui2mapped_pheno[cid] in pheno2pub_fetched:
                continue
            fout.write('\t'.join([cui2mapped_pheno[cid]] + pmids[:MAX_PMID_NUM]) + '\n')
            pheno2pub_fetched[cui2mapped_pheno[cid]] = pmids
        
        for i, term in enumerate(target_phenos):
            if term in pheno2pub_fetched:
                continue

            if (i+1) % SLEEP_INTERVAL == 0:
                time.sleep(float(REST_SEC)) 

            pubmed_ids = fetch_pubmed_ids(term, max_results=MAX_PMID_NUM)
            if not pubmed_ids:
                parts = term.split(', ')
                if len(parts) > 1:
                    sub_terms = list(set([parts[0].strip(), ', '.join([parts[0].strip(), parts[1].strip()]), ', '.join([parts[0].strip(), parts[-1].strip()])]))
                    pmid_sub = []
                    for sub in sub_terms:
                        print(sub)
                        pmid_sub.extend(fetch_pubmed_ids(sub, max_results=5))
                    pubmed_ids = list(set(pmid_sub))
            
            fout.write('\t'.join([term]+ pubmed_ids) + '\n')
            # print(f"PubMed IDs for {term}: {', '.join(pubmed_ids)}")
            pheno2pub_fetched[term] = pubmed_ids

    with open(data_root / (fname_out.split('.')[0]) + '.json', 'w') as f:
        json.dump(pheno2pub_fetched, f, indent=2)