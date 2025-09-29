# :sparkles: DIVA: Disease-specific variant pathogenicity prediction using multimodal biomedical language models

We released our predictions at [https://diva.yulab.org/](https://diva.yulab.org/)


## Dependencies

```
biopython==1.79
numpy==1.24.4
pandas==1.4.4
PyYAML==6.0
requests==2.32.5
scikit_learn==1.3.2
scipy==1.6.3
torch==1.9.0
torch==1.13.1+cu117
transformers==4.28.1
```
See `requirements.txt`

## Prepare input data
Example inputs are provided in `example_data/`

### Prepare variant inputs

* Context variants (required for inference): known disease variants that provide local disease contextual knowledge
* Variants of interest for prediction

### Prepare input feature for protein
* Sequence file (FASTA)
* Function annotation file

### Prepare disease vocabulary
* Vocabulary file
* Disease annotation file
* (Optional) Disease name mapping file

### AlphaMissense data (optional)

* Download AlphaMissense predictions (`AlphaMissense_aa_substitutions.tsv.gz`) from [here](https://console.cloud.google.com/storage/browser/dm_alphamissense)
* Process with `preprocess/prep_alphamissense.py`
* *Set `use_alphamissense=False` in CONFIG if choosing not to use AlphaMissense prediction scores*

## Running DIVA

The following scripts are intended to be run from the `dev/` directory.

### Model training

```
python -u disease_model_pipeline.py --config ./configs/dis_var_train_config.yaml --tensorboard True
```


### Inference

```
python -u disease_inference.py --config ./configs/dis_var_pred_config.yaml
```

### Inference example

```
python -u disease_inference.py --config ./configs/example_pred.yaml
```

## Reference
Liu, Yilin, David N. Cooper, and Haiyuan Yu. "Disease-specific variant pathogenicity prediction using multimodal biomedical language models." [bioRxiv (2025)](https://www.biorxiv.org/content/10.1101/2025.09.09.675184v1)