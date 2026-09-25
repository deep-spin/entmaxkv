from entmaxkv.kv_cache import PagedKVCache
from entmaxkv.attention_topk import (
    sparse_attention_decode_paged,
)
from entmaxkv.attention_gaussian import (
    sparse_attention_decode_gaussian_aware_entmax,
)
from entmaxkv.selectors import (
    GaussianPageSelector,
    SelectedPages,
    TopKPageSelector,
)

__version__ = "0.2.0"

__all__ = [
    "PagedKVCache",
    "sparse_attention_decode_paged",
    "sparse_attention_decode_gaussian_aware_entmax",
    "GaussianPageSelector",
    "SelectedPages",
    "TopKPageSelector",
]
