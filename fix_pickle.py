import numpy as np
from sklearn.tree._tree import _check_node_ndarray, NODE_DTYPE

original_check = _check_node_ndarray

def patched_check(node_ndarray, expected_dtype=NODE_DTYPE):
    if node_ndarray.dtype.names is not None and len(node_ndarray.dtype.names) == 7:
        expected_names = expected_dtype.names if hasattr(expected_dtype, 'names') else list(expected_dtype.fields.keys())
        new_arr = np.empty(node_ndarray.shape, dtype=expected_dtype)
        for name in node_ndarray.dtype.names:
            new_arr[name] = node_ndarray[name]
        new_arr['missing_go_to_left'] = 0
        node_ndarray = new_arr
    return original_check(node_ndarray, expected_dtype)

import sklearn.tree._tree as _tree
_tree._check_node_ndarray = patched_check

import joblib
model = joblib.load('fake_news_model.pkl')
print('Model loaded successfully:', type(model).__name__)

joblib.dump(model, 'fake_news_model.pkl')
print('Model re-saved successfully!')
