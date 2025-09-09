# DIVA: Disease-specific variant pathogenicity prediction using multimodal biomedical language models

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

## AlphaMissense data

* Download AlphaMissense predictions (`AlphaMissense_aa_substitutions.tsv.gz`) from [here](https://console.cloud.google.com/storage/browser/dm_alphamissense)
* Process with `preprocess/prep_alphamissense.py`

## Running DIVA

### Model training

```
python -u disease_model_pipeline.py --config ./configs/dis_var_config.yaml --tensorboard True
```


### Inference

```
python -u disease_inference.py --config ./configs/dis_var_pred_config.yaml
```

## Reference
