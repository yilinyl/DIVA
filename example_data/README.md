# Example data for inference


### Variant inputs

* Context variants: `example_context_vars.csv`
* Variants of interest for prediction: `example_vars.csv`

Each variant file should at least include:
* Protein ID (`UniProt`)
* Variant position on protein (`Protein_position`)
* wild-type (`REF_AA`), alternate amino acids (`ALT_AA`)
* Disease condition (`phenotype`)
  - For variants without known disease association (e.g. VUS), this can be a presumed disease based on availble evidence, or it may be randomly imputed. The model would still generate a ranked list of likely associated diseases among the provided vocabulary.
* Label of deleteriousness: `benign=0, pathogenic=1, VUS=-1`


### Input feature for protein
* Sequence file (FASTA): `example.fasta`
* Function annotation file: `protein_desc.txt`

### Disease vocabulary
* Vocabulary file: `example_disease_vocab.txt`
* Disease annotation file: `example_disease_desc.json`
