import os, sys
from pathlib import Path
import re
import json
from collections import defaultdict
from owlready2 import get_ontology, ThingClass


def get_parents(term):
    parents = []
    for parent_node in term.is_a:
        if isinstance(parent_node, ThingClass):
            parents.append(parent_node.name.replace('_', ':'))
    return parents


def get_children(term):
    children = []
    for child_node in term.subclasses():
        if isinstance(child_node, ThingClass):
            children.append(child_node.name.replace('_', ':'))
    return children


if __name__ == '__main__':
    mondo_root = Path('/home/yl986/data/variant_data/disease_ontology/mondo/')
    mondo_fpath = mondo_root / 'mondo.owl'
    parsed_id_fname = 'term2mondo_id.json'
    result_fname = 'mondo_parsed_dict.json'
    mondo = get_ontology(str(mondo_fpath)).load()
    print('Mondo ontology loaded')

    disname2id = dict()
    complete_info_dict = dict()
    # if (mondo_root / parsed_id_fname).exists():
    #     with open(mondo_root / parsed_id_fname) as f:
    #         disname2id = json.load(f)
    #     print('{} terms loaded from cache'.format(len(disname2id)))
    exclude_branch = ['http://purl.obolibrary.org/obo/MONDO_0005583']  # non-human animal disease
    mondo_parsed_info = []
    for i, disease in enumerate(mondo.classes()):
        if (i+1) % 1000 == 0:
            print(i+1, 'processed')
            # save 
            with open(mondo_root / parsed_id_fname, 'w') as f:
                json.dump(disname2id, f, indent=2)

            with open(mondo_root / result_fname, 'w') as f:
                json.dump(complete_info_dict, f, indent=2)
        
        if not isinstance(disease, ThingClass):
            continue
        if not disease.name.startswith('MONDO'):
            continue
        
        if any([s.iri in exclude_branch for s in disease.ancestors()]):
            print('Exclude non-human disease:', disease.name)
            continue

        mondo_id = disease.name.replace('_', ':')
        try:
            name = disease.label[0].lower()
        except IndexError:
            print(disease)
            continue
        
        if disease.IAO_0000115:  # disease definition
            desc = disease.IAO_0000115[0]
        else:
            desc = ''
    
        disname2id[name] = mondo_id
        # for exact_syn in disease.hasExactSynonym:
        #     disname2id[exact_syn.lower()] = mondo_id
        # for rel_syn in disease.hasRelatedSynonym:
        #     disname2id[rel_syn.lower()] = mondo_id
        # for broad_syn in disease.hasBroadSynonym:
        #     disname2id[rel_syn.lower()] = mondo_id
        exact_syn = disease.hasExactSynonym
        other_syn = disease.hasRelatedSynonym + disease.hasBroadSynonym + disease.hasNarrowSynonym
        for syn in exact_syn + other_syn:
            disname2id[syn.lower()] = mondo_id
        try:
            complete_info_dict[mondo_id] = {'name': name,
                                            'def': desc,
                                            'exact_syn': exact_syn,  # exact synonym
                                            'other_syn': other_syn,  # other synonym
                                            'parent': get_parents(disease),
                                            'child': get_children(disease)}
        except:
            print(disease)
            continue

    print('Parsing complete! Save file...')
    with open(mondo_root / parsed_id_fname, 'w') as f:
        json.dump(disname2id, f, indent=2)

    with open(mondo_root / result_fname, 'w') as f:
        json.dump(complete_info_dict, f, indent=2)
    