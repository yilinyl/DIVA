from .han import HAN
from .graphtransformer_v0 import GraphTransformer

__model_lib = {
    'han': HAN,
    'gtn': GraphTransformer
}

def build_model(model='', **kwargs):
    model = model.lower()
    avail_models = list(__model_lib.keys())

    if model not in avail_models:
        raise KeyError(f'Unknown model: {model}. Must be one of {avail_models}')
    return __model_lib[model](**kwargs)
