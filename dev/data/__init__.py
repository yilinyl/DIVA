from .datasets import VariantSeqGraphDataSet
from .dataset_multi import MultiModalDataSet

__data_lib = {
    'seq': VariantSeqGraphDataSet,
    'multi': MultiModalDataSet,
}

def build_dataset(dtype, df_in, sift_map=None, **kwargs):
    dtype = dtype.lower()
    avail_dtypes = list(__data_lib.keys())

    if dtype not in avail_dtypes:
        raise KeyError(f'Unknown data type: {dtype}. Must be one of {avail_dtypes}')
    
    return __data_lib[dtype](df_in, sift_map=sift_map, **kwargs)