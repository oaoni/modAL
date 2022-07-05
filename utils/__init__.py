from .combination import make_linear_combination, make_product, make_query_strategy
from .data import data_vstack, addData2matrix
from .selection import multi_argmax, weighted_random
from .validation import check_class_labels, check_class_proba
from .query import policy_func, active_sample, max_uncertainty, weighted_uncertainty,\
                   random_query, leverage_online, leverage_observed, max_guided_density,\
                   weighted_guided_density, max_guided_diversity, guided_exploration, max_unique_uncertainty

__all__ = [
    'make_linear_combination', 'make_product', 'make_query_strategy',
    'data_vstack','addData2matrix',
    'multi_argmax', 'weighted_random',
    'check_class_labels', 'check_class_proba',
    'policy_func','active_sample', 'max_uncertainty','weighted_uncertainty','random_query',
    'leverage_online', 'leverage_observed', 'max_guided_density', 'weighted_guided_density',
    'max_guided_diversity', 'guided_exploration', 'max_unique_uncertainty'
]
