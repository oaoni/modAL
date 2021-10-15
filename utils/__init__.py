from .combination import make_linear_combination, make_product, make_query_strategy
from .data import data_vstack, addData2matrix
from .selection import multi_argmax, weighted_random
from .validation import check_class_labels, check_class_proba
from .query import active_sample, max_uncertainty, weighted_uncertainty, random_query, leverage_online, leverage_observed

__all__ = [
    'make_linear_combination', 'make_product', 'make_query_strategy',
    'data_vstack','addData2matrix',
    'multi_argmax', 'weighted_random',
    'check_class_labels', 'check_class_proba',
    'active_sample', 'max_uncertainty','weighted_uncertainty','random_query',
    'leverage_online', 'leverage_observed'
]
