"""
modified from goatools GitHub release
"""

from __future__ import print_function

import sys
import signal
from collections import Counter
from collections import defaultdict
import numpy as np
from goatools import semantic

# from goatools.godag.consts import NAMESPACE2GO
# from goatools.godag.consts import NAMESPACE2NS
from goatools.godag.go_tasks import get_go2ancestors
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.godag.relationship_combos import RelationshipCombos
from goatools.anno.update_association import clean_anno
from goatools.utils import get_b2aset


NAMESPACE2ID = {'human_phenotype': 'HP:0000001'}

# Define a handler that raises a TimeoutError when the signal is received
def timeout_handler(signum, frame):
    raise TimeoutError(f"Function execution exceeded {signum} seconds.")


def run_with_timeout(func, signum, *args, **kwargs):
    # Start an alarm to go off after 5 seconds
    signal.alarm(signum)
    try:
        result = func(*args, **kwargs)
    finally:
        # Disable the alarm after function execution
        signal.alarm(0)
    return result


class TermCountsHPO:
    '''
        TermCounts counts the term counts for each
    '''
    # pylint: disable=too-many-instance-attributes
    def __init__(self, go2obj, annots, relationships=None, **kws):
        '''
            Initialise the counts and
        '''
        _prt = kws.get('prt')
        # Backup
        self.go2obj = go2obj  # Full GODag
        self.annots, go_alts = clean_anno(annots, go2obj, _prt)[:2]
        # Genes annotated to all associated GO, including inherited up ancestors'
        _relationship_set = RelationshipCombos(go2obj).get_set(relationships)
        self.go2genes = self._init_go2genes(_relationship_set, go2obj)
        self.gene2gos = get_b2aset(self.go2genes)
        # Annotation main GO IDs (prefer main id to alt_id)
        self.goids = set(self.go2genes.keys())
        self.gocnts = Counter({go:len(geneset) for go, geneset in self.go2genes.items()})
        # Get total count for each branch: BP MF CC
        self.aspect_counts = {
            'human_phenotype': self.gocnts.get(NAMESPACE2ID['human_phenotype'], 0)}  # HPO
        
        self._init_add_goid_alt(go_alts)
        self.gosubdag = GoSubDag(
            set(self.gocnts.keys()),
            go2obj,
            tcntobj=self,
            relationships=_relationship_set,
            prt=None)
        if _prt:
            self.prt_objdesc(_prt)

    def get_annotations_reversed(self):
        """Return go2geneset for all GO IDs explicitly annotated to a gene"""
        return set.union(*get_b2aset(self.annots))

    def _init_go2genes(self, relationship_set, godag):
        '''
            Fills in the genes annotated to each GO, including ancestors

            Due to the ontology structure, gene products annotated to
            a GO Terma are also annotated to all ancestors.
        '''
        go2geneset = defaultdict(set)
        go2up = get_go2ancestors(set(godag.values()), relationship_set)
        # Fill go-geneset dict with GO IDs in annotations and their corresponding counts
        for geneid, goids_anno in self.annots.items():
            # Make a union of all the terms for a gene, if term parents are
            # propagated but they won't get double-counted for the gene
            allterms = set()
            for goid_main in goids_anno:
                allterms.add(goid_main)
                if goid_main in go2up:
                    allterms.update(go2up[goid_main])
            # Add 1 for each GO annotated to this gene product
            for ancestor in allterms:
                go2geneset[ancestor].add(geneid)
        return dict(go2geneset)

    def _init_add_goid_alt(self, not_main):
        '''
            Add alternate GO IDs to term counts. Report GO IDs not found in GO DAG.
        '''
        if not not_main:
            return
        for go_id in not_main:
            if go_id in self.go2obj:
                goid_main = self.go2obj[go_id].item_id
                self.gocnts[go_id] = self.gocnts[goid_main]
                self.go2genes[go_id] = self.go2genes[goid_main]

    def get_count(self, go_id):
        '''
            Returns the count of that GO term observed in the annotations.
        '''
        return self.gocnts[go_id]

    def get_total_count(self, aspect):
        '''
            Gets the total count that's been precomputed.
        '''
        return self.aspect_counts[aspect]

    def get_term_freq(self, go_id):
        '''
            Returns the frequency at which a particular GO term has
            been observed in the annotations.
        '''
        num_ns = float(self.get_total_count(self.go2obj[go_id].namespace))
        return float(self.get_count(go_id))/num_ns if num_ns != 0 else 0


def get_info_content(go_id, termcounts):
    '''
        Retrieve the information content of a GO term.
    '''
    ntd = termcounts.gosubdag.go2nt.get(go_id)
    return ntd.tinfo if ntd else 0.0


def agg_lin_similarity(term_set1, term_set2, onto_dag, termcounts):
    # 
    """
        Semantic similarities of multiple terms (e.g. gene-level similarity by GO terms)
        - Aggregation: best-match average
        - Similarity measure: Lin similarity
    """
    if not isinstance(term_set1, set):
        term_set1 = set(term_set1)
    if not isinstance(term_set2, set):
        term_set2 = set(term_set2)
    best_sims = []
    for tid1 in term_set1:
        if tid1 in term_set2:
            best_sims.append(1)
        else:
            best_sims.append(max(semantic.lin_sim(tid1, tid2, onto_dag, termcounts) for tid2 in term_set2))
    for tid2 in term_set2:
        if tid2 in term_set1:
            best_sims.append(1)
        else:
            best_sims.append(max(semantic.lin_sim(tid1, tid2, onto_dag, termcounts) for tid1 in term_set1))

    return np.mean(best_sims)

