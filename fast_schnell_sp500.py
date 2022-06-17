import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm.autonotebook import tqdm
from sklearn.metrics import silhouette_score
from sklearn.base import ClusterMixin, TransformerMixin
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import warnings
from scipy.sparse import issparse
from scipy import linalg, sparse
from sklearn.utils.estimator_checks import _NotAnArray as NotAnArray
from sklearn.utils import check_random_state
from scipy.spatial.distance import cdist
from sklearn.cluster._kmeans import _kmeans_plusplus
from scipy.interpolate import interp1d
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import adfuller

import yfinance as yf
import requests
import yahoo_fin.stock_info as si
import bs4 as bs
import pickle

#Without usage of the import yahoo_fin.stock_info as si sp500 = si.tickers_sp500()
# def save_sp500_tickers():
#     resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
#     soup = bs.BeautifulSoup(resp.text, 'lxml')
#     table = soup.find('table', {'class': 'wikitable sortable'})
#     tickers = []
#     for row in table.findAll('tr')[1:]:
#         ticker = row.findAll('td')[0].text
#         tickers.append(ticker[:-1])

#     with open("sp500tickers.pickle","wb") as f:
#         pickle.dump(tickers,f)
        
#     tik = []
#     for t in tickers:
#         t = t.replace('.', '-')
#         tik.append(t)

#     return tik

# tickers = save_sp500_tickers()

def download_and_preprocess_data(tickers, start, end, interval):
    df = yf.download(tickers = tickers, start = start, end = end, interval = interval)
    df = df.drop(['Adj Close', 'Open', 'High', 'Low', 'Volume'], axis=1) #we work with Close prices
    df = df.droplevel(level=0, axis=1)
    df = df.bfill()
    df = df.dropna(axis='columns')
    return df

def interactive_plot(df, title):
    fig = px.line(title = title)
    for i in df.columns[1:]:
        fig.add_scatter(x = df.index, y = df[i], name = i)
    fig.show()
#Matplotlib works faster but the plot is not that clear:
#     plt.figure(figsize=(120,80))   
#     for column in df.columns:
#         plt.plot(df[column], label=column, linewidth=6)
#     plt.title(title)
#     plt.legend(loc='upper right')
#     plt.show()

def normalize(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i] = x[i]/x[i][0]
    return x

def ticker_info():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    industry = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker[:-1])
        
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[3].text
        industry.append(ticker[:-1])

    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    tik = []
    for t in tickers:
        t = t.replace('.', '-')
        tik.append(t)
    tik = pd.DataFrame(tik, columns=['Ticker'])
    industry = pd.DataFrame(industry, columns=['Industry'])
    return pd.concat([tik, industry], axis=1)

def normalized_plots(df_ticker_info, stocks):
    print("=================================================================================")
    print(f"                          Normalized plots for every industry")

    ind = set(df_ticker_info['Industry'])

    for i in ind:
        globals()[f"cluster_{i}"] = []
        for k in range(0,len(df_ticker_info['Ticker'])):
            if df_ticker_info.iloc[k]['Industry'] == i and df_ticker_info.iloc[k][0] in stocks.columns:
                globals()[f"cluster_{i}"].append(stocks[df_ticker_info.iloc[k]['Ticker']])
        globals()[f"cluster_{i}"] = (pd.DataFrame(globals()[f"cluster_{i}"]))
        if globals()[f"cluster_{i}"].empty == False:
            print("=================================================================================")
            print(f"                          Industry: {i}")
            print("=================================================================================")
            fig = px.line()
            for column in globals()[f"cluster_{i}"].T.columns:
                fig.add_scatter(x = globals()[f"cluster_{i}"].T.index, y = globals()[f"cluster_{i}"].T[column], name = column)
            fig.show()

    #     plt.figure(figsize=(12,8))
    #     for column in globals()[f"cluster_{i}"].T.columns:
    #         plt.plot(normalize(globals()[f"cluster_{i}"].T))
        #plt.legend()
        plt.show()

def add_industry(df_ticker_info, stocks):
    df_ticker_info.index = df_ticker_info['Ticker']
    df_ticker_info['Ticker'] = df_ticker_info.index
    snp500 = pd.concat([df_ticker_info, stocks.T], axis=1)
    
    column_ = []
    for i in range(0,len(snp500.columns)):
        if len(str(snp500.columns[i]))>18:
            column_.append(str(snp500.columns[i])[0:10])
        else:
            column_.append(str(snp500.columns[i]))
    
    snp500.columns = column_
    
    return snp500


#Based on https://github.com/tslearn-team/tslearn,
#https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html,
#https://readthedocs.org/projects/tslearn/downloads/pdf/latest/.

    
    
    
    
    class TimeSeriesCentroidBasedClusteringMixin:

        def _post_fit(self, X_fitted, centroids, inertia):
            if np.isfinite(inertia) and (centroids is not None):
                self.cluster_centers_ = centroids
                self._assign(X_fitted)
                self._X_fit = X_fitted
                self.inertia_ = inertia
            else:
                self._X_fit = None
                
#Using several files from tslearn package:
class TimeSeriesCentroidBasedClusteringMixin:

    def _post_fit(self, X_fitted, centroids, inertia):
        if np.isfinite(inertia) and (centroids is not None):
            self.cluster_centers_ = centroids
            self._assign(X_fitted)
            self._X_fit = X_fitted
            self.inertia_ = inertia
        else:
            self._X_fit = None

class BaseModelPackage:
    __metaclass__ = ABCMeta

    @abstractmethod
    def _is_fitted(self):
        pass

    def _get_model_params(self):
        params = {}
        for attr in dir(self):
            # Do not save properties
            if (hasattr(type(self), attr) and
                    isinstance(getattr(type(self), attr), property)):
                continue
            if (not attr.startswith("__") and
                    attr.endswith("_") and
                    not callable(getattr(self, attr))):
                params[attr] = getattr(self, attr)
        return params

    def _to_dict(self, output=None, hyper_parameters_only=False):

        if not self._is_fitted():
            raise NotFittedError("Model must be fit before it can be packaged")

        d = {'hyper_params': self.get_params(),
             'model_params': self._get_model_params()}

        # This is just for json support to convert numpy arrays to lists
        if output == 'json':
            d['model_params'] = BaseModelPackage._listify(d['model_params'])
            d['hyper_params'] = BaseModelPackage._listify(d['hyper_params'])

        elif output == 'hdf5':
            d['hyper_params'] = \
                BaseModelPackage._none_to_str(d['hyper_params'])

        if hyper_parameters_only:
            del d["model_params"]

        return d

    @staticmethod
    def _none_to_str(mp):
        """Use str to store Nones. Used for HDF5"""
        for k in mp.keys():
            if mp[k] is None:
                mp[k] = 'None'

        return mp

    @staticmethod
    def _listify(model_params):
        for k in model_params.keys():
            param = model_params[k]

            if isinstance(param, np.ndarray):
                model_params[k] = param.tolist()  # for json support
            elif isinstance(param, list) and isinstance(param[0], np.ndarray):
                model_params[k] = [p.tolist() for p in param]  # json support
            else:
                model_params[k] = param
        return model_params

    @staticmethod
    def _organize_model(cls, model):

        model_params = model.pop('model_params')
        hyper_params = model.pop('hyper_params')  # hyper-params

        # instantiate with hyper-parameters
        inst = cls(**hyper_params)

        # set all model params
        for p in model_params.keys():
            setattr(inst, p, model_params[p])

        return inst

    @classmethod
    def _byte2string(cls, model):
        for param_set in ['hyper_params', 'model_params']:
            for k in model[param_set].keys():
                if type(model[param_set][k]) == type(b''):
                    model[param_set][k] = model[param_set][k].decode('utf-8')
        return model


    def to_hdf5(self, path):
        if not HDF5_INSTALLED:
            raise ImportError(h5py_msg)

        d = self._to_dict(output='hdf5')
        hdftools.save_dict(d, path, 'data')

    @classmethod
    def from_hdf5(cls, path):
        if not HDF5_INSTALLED:
            raise ImportError(h5py_msg)

        model = hdftools.load_dict(path, 'data')
        model = cls._byte2string(model)

        for k in model['hyper_params'].keys():
            if model['hyper_params'][k] == 'None':
                model['hyper_params'][k] = None

        return cls._organize_model(cls, model)

    def to_json(self, path):

        d = self._to_dict(output='json')
        json.dump(d, open(path, 'w'))

    @classmethod
    def from_json(cls, path):

        model = json.load(open(path, 'r'))
        model = cls._byte2string(model)

        # Convert the lists back to arrays
        for param_type in ['model_params', 'hyper_params']:
            for k in model[param_type].keys():
                param = model[param_type][k]
                if type(param) is list:
                    arr = np.array(param)
                    if arr.dtype == object:
                        # Then maybe it was rather a list of arrays
                        # This is very hacky...
                        arr = [np.array(p) for p in param]
                    model[param_type][k] = arr

        return cls._organize_model(cls, model)

    def to_pickle(self, path):

        d = self._to_dict()
        pickle.dump(d, open(path, 'wb'), protocol=2)

    @classmethod
    def from_pickle(cls, path):
        model = pickle.load(open(path, 'rb'))
        model = cls._byte2string(model)
        return cls._organize_model(cls, model)

class TimeSeriesBaseEstimator(BaseEstimator):
    def _more_tags(self):
        return _DEFAULT_TAGS


class BaseModelPackage:
    __metaclass__ = ABCMeta

    @abstractmethod
    def _is_fitted(self):
        pass

    def _get_model_params(self):
        """Get model parameters that are sufficient to recapitulate it."""
        params = {}
        for attr in dir(self):
            # Do not save properties
            if (hasattr(type(self), attr) and
                    isinstance(getattr(type(self), attr), property)):
                continue
            if (not attr.startswith("__") and
                    attr.endswith("_") and
                    not callable(getattr(self, attr))):
                params[attr] = getattr(self, attr)
        return params

    def _to_dict(self, output=None, hyper_parameters_only=False):

        if not self._is_fitted():
            raise NotFittedError("Model must be fit before it can be packaged")

        d = {'hyper_params': self.get_params(),
             'model_params': self._get_model_params()}

        # This is just for json support to convert numpy arrays to lists
        if output == 'json':
            d['model_params'] = BaseModelPackage._listify(d['model_params'])
            d['hyper_params'] = BaseModelPackage._listify(d['hyper_params'])

        elif output == 'hdf5':
            d['hyper_params'] = \
                BaseModelPackage._none_to_str(d['hyper_params'])

        if hyper_parameters_only:
            del d["model_params"]

        return d

    @staticmethod
    def _none_to_str(mp):
        """Use str to store Nones. Used for HDF5"""
        for k in mp.keys():
            if mp[k] is None:
                mp[k] = 'None'

        return mp

    @staticmethod
    def _listify(model_params):

        for k in model_params.keys():
            param = model_params[k]

            if isinstance(param, np.ndarray):
                model_params[k] = param.tolist()  # for json support
            elif isinstance(param, list) and isinstance(param[0], np.ndarray):
                model_params[k] = [p.tolist() for p in param]  # json support
            else:
                model_params[k] = param
        return model_params

    @staticmethod
    def _organize_model(cls, model):

        model_params = model.pop('model_params')
        hyper_params = model.pop('hyper_params')  # hyper-params

        # instantiate with hyper-parameters
        inst = cls(**hyper_params)

        # set all model params
        for p in model_params.keys():
            setattr(inst, p, model_params[p])

        return inst

    @classmethod
    def _byte2string(cls, model):
        for param_set in ['hyper_params', 'model_params']:
            for k in model[param_set].keys():
                if type(model[param_set][k]) == type(b''):
                    model[param_set][k] = model[param_set][k].decode('utf-8')
        return model


    def to_hdf5(self, path):

        if not HDF5_INSTALLED:
            raise ImportError(h5py_msg)

        d = self._to_dict(output='hdf5')
        hdftools.save_dict(d, path, 'data')

    @classmethod
    def from_hdf5(cls, path):

        if not HDF5_INSTALLED:
            raise ImportError(h5py_msg)

        model = hdftools.load_dict(path, 'data')
        model = cls._byte2string(model)

        for k in model['hyper_params'].keys():
            if model['hyper_params'][k] == 'None':
                model['hyper_params'][k] = None

        return cls._organize_model(cls, model)

    def to_json(self, path):

        d = self._to_dict(output='json')
        json.dump(d, open(path, 'w'))

    @classmethod
    def from_json(cls, path):

        model = json.load(open(path, 'r'))
        model = cls._byte2string(model)

        # Convert the lists back to arrays
        for param_type in ['model_params', 'hyper_params']:
            for k in model[param_type].keys():
                param = model[param_type][k]
                if type(param) is list:
                    arr = np.array(param)
                    if arr.dtype == object:
                        # Then maybe it was rather a list of arrays
                        # This is very hacky...
                        arr = [np.array(p) for p in param]
                    model[param_type][k] = arr

        return cls._organize_model(cls, model)

    def to_pickle(self, path):

        d = self._to_dict()
        pickle.dump(d, open(path, 'wb'), protocol=2)

    @classmethod
    def from_pickle(cls, path):
        model = pickle.load(open(path, 'rb'))
        model = cls._byte2string(model)
        return cls._organize_model(cls, model)

def to_time_series(ts, remove_nans=False):
    ts_out = np.array(ts, copy=True)
    if ts_out.ndim <= 1:
        ts_out = ts_out.reshape((-1, 1))
    if ts_out.dtype != float:
        ts_out = ts_out.astype(float)
    if remove_nans:
        ts_out = ts_out[:ts_size(ts_out)]
    return ts_out

def ts_size(ts):
    ts_ = to_time_series(ts)
    sz = ts_.shape[0]
    while sz > 0 and np.all(np.isnan(ts_[sz - 1])):
        sz -= 1
    return sz

def to_time_series_dataset(dataset, dtype=float):
    try:
        #import pandas as pd
        if isinstance(dataset, pd.DataFrame):
            return to_time_series_dataset(np.array(dataset))
    except ImportError:
        pass
    if isinstance(dataset, NotAnArray):  # Patch to pass sklearn tests
        return to_time_series_dataset(np.array(dataset))
    if len(dataset) == 0:
        return np.zeros((0, 0, 0))
    if np.array(dataset[0]).ndim == 0:
        dataset = [dataset]
    n_ts = len(dataset)
    max_sz = max([ts_size(to_time_series(ts, remove_nans=True))
                  for ts in dataset])
    d = to_time_series(dataset[0]).shape[1]
    dataset_out = np.zeros((n_ts, max_sz, d), dtype=dtype) + np.nan
    for i in range(n_ts):
        ts = to_time_series(dataset[i], remove_nans=True)
        dataset_out[i, :ts.shape[0]] = ts
    return dataset_out.astype(dtype)

def _check_initial_guess(init, n_clusters):
    if hasattr(init, '__array__'):
        assert init.shape[0] == n_clusters, \
            "Initial guess index array must contain {} samples," \
            " {} given".format(n_clusters, init.shape[0])

class EmptyClusterError(Exception):
    def __init__(self, message=""):
        super().__init__()
        self.message = message

    def __str__(self):
        if len(self.message) > 0:
            suffix = " (%s)" % self.message
        else:
            suffix = ""
        return "Cluster assignments lead to at least one empty cluster" + \
               suffix

def check_equal_size(dataset):
    dataset_ = to_time_series_dataset(dataset)
    if len(dataset_) == 0:
        return True

    size = ts_size(dataset[0])
    return all(ts_size(ds) == size for ds in dataset_[1:])

def check_dims(X, X_fit_dims=None, extend=True, check_n_features_only=False):
    if X is None:
        raise ValueError('X is equal to None!')

    if extend and len(X.shape) == 2:
        warnings.warn('2-Dimensional data passed. Assuming these are '
                      '{} 1-dimensional timeseries'.format(X.shape[0]))
        X = X.reshape((X.shape) + (1,))

    if X_fit_dims is not None:
        if check_n_features_only:
            if X_fit_dims[2] != X.shape[2]:
                raise ValueError(
                    'Number of features of the provided timeseries'
                    '(last dimension) must match the one of the fitted data!'
                    ' ({} and {} are passed shapes)'.format(X_fit_dims,
                                                            X.shape))
        else:
            if X_fit_dims[1:] != X.shape[1:]:
                raise ValueError(
                    'Dimensions of the provided timeseries'
                    '(except first) must match those of the fitted data!'
                    ' ({} and {} are passed shapes)'.format(X_fit_dims,
                                                            X.shape))

    return X

class TimeSeriesResampler(TransformerMixin):

    def __init__(self, sz):
        self.sz_ = sz

    def fit(self, X, y=None, **kwargs):

        return self

    def _transform_unit_sz(self, X):
        n_ts, sz, d = X.shape
        X_out = np.empty((n_ts, self.sz_, d))
        for i in range(X.shape[0]):
            X_out[i] = np.nanmean(X[i], axis=0, keepdims=True)
        return X_out

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X).transform(X)

    def transform(self, X, y=None, **kwargs):

        X_ = to_time_series_dataset(X)
        if self.sz_ == 1:
            return self._transform_unit_sz(X_)
        n_ts, sz, d = X_.shape
        equal_size = check_equal_size(X_)
        X_out = np.empty((n_ts, self.sz_, d))
        for i in range(X_.shape[0]):
            xnew = np.linspace(0, 1, self.sz_)
            if not equal_size:
                sz = ts_size(X_[i])
            for di in range(d):
                f = interp1d(np.linspace(0, 1, sz), X_[i, :sz, di],
                             kind="slinear")
                X_out[i, :, di] = f(xnew)
        return X_out


class TimeSeriesScalerMinMax(TransformerMixin, TimeSeriesBaseEstimator):

    def __init__(self, value_range=(0., 1.)):
        self.value_range = value_range

    def fit(self, X, y=None, **kwargs):

        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = to_time_series_dataset(X)
        self._X_fit_dims = X.shape
        return self

    def fit_transform(self, X, y=None, **kwargs):

        return self.fit(X).transform(X)

    def transform(self, X, y=None, **kwargs):

        value_range = self.value_range

        if value_range[0] >= value_range[1]:
            raise ValueError("Minimum of desired range must be smaller"
                             " than maximum. Got %s." % str(value_range))

        check_is_fitted(self, '_X_fit_dims')
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X_ = to_time_series_dataset(X)
        X_ = check_dims(X_, X_fit_dims=self._X_fit_dims, extend=False)
        min_t = np.nanmin(X_, axis=1)[:, np.newaxis, :]
        max_t = np.nanmax(X_, axis=1)[:, np.newaxis, :]
        range_t = max_t - min_t
        range_t[range_t == 0.] = 1.
        nomin = (X_ - min_t) * (value_range[1] - value_range[0])
        X_ = nomin / range_t + value_range[0]
        return X_

    def _more_tags(self):
        return {'allow_nan': True}

def _check_full_length(centroids):

    resampler = TimeSeriesResampler(sz=centroids.shape[1])
    return resampler.fit_transform(centroids)

def _check_no_empty_cluster(labels, n_clusters):

    for k in range(n_clusters):
        if np.sum(labels == k) == 0:
            raise EmptyClusterError

def _compute_inertia(distances, assignments, squared=True):

    n_ts = distances.shape[0]
    if squared:
        return np.sum(distances[np.arange(n_ts),
                                   assignments] ** 2) / n_ts
    else:
        return np.sum(distances[np.arange(n_ts), assignments]) / n_ts

def _set_weights(w, n):
    if w is None or len(w) != n:
        w = np.ones((n, ))
    return w

def euclidean_barycenter(X, weights=None):

    X_ = to_time_series_dataset(X)
    weights = _set_weights(weights, X_.shape[0])
    return np.average(X_, axis=0, weights=weights)

def _k_init_metric(X, n_clusters, cdist_metric, random_state,
                   n_local_trials=None):

    n_samples, n_timestamps, n_features = X.shape

    centers = np.empty((n_clusters, n_timestamps, n_features),
                          dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = cdist_metric(centers[0, np.newaxis], X) ** 2
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                           rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                   out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = cdist_metric(X[candidate_ids], X) ** 2

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                      out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]

    return centers

def _return_float_dtype(X, Y):
    """
    1. If dtype of X and Y is float32, then dtype float32 is returned.
    2. Else dtype float is returned.
    """
    if not issparse(X) and not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if Y is None:
        Y_dtype = X.dtype
    elif not issparse(Y) and not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
        Y_dtype = Y.dtype
    else:
        Y_dtype = Y.dtype

    if X.dtype == Y_dtype == np.float32:
        dtype = np.float32
    else:
        dtype = float

    return X, Y, dtype

def check_pairwise_arrays(X, Y, precomputed=False, dtype=None):
    X, Y, dtype_float = _return_float_dtype(X, Y)

    #warn_on_dtype = dtype is not None
    estimator = 'check_pairwise_arrays'
    if dtype is None:
        dtype = dtype_float

    if Y is X or Y is None:
        X = Y = check_array(X, accept_sparse='csr', dtype=dtype,
                            estimator=estimator)
    else:
        X = check_array(X, accept_sparse='csr', dtype=dtype,
                        estimator=estimator)
        Y = check_array(Y, accept_sparse='csr', dtype=dtype,
                        estimator=estimator)

    if precomputed:
        if X.shape[1] != Y.shape[0]:
            raise ValueError("Precomputed metric requires shape "
                             "(n_queries, n_indexed). Got (%d, %d) "
                             "for %d indexed." %
                             (X.shape[0], X.shape[1], Y.shape[0]))
    elif X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             X.shape[1], Y.shape[1]))

    return X, Y

def row_norms(X, squared=False):

    if sparse.issparse(X):
        if not isinstance(X, sparse.csr_matrix):
            X = sparse.csr_matrix(X)
        norms = csr_row_norms(X)
    else:
        norms = np.einsum('ij,ij->i', X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms

def safe_sparse_dot(a, b, dense_output=False):
    if sparse.issparse(a) or sparse.issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)

def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
                        X_norm_squared=None):
    X, Y = check_pairwise_arrays(X, Y)

    if X_norm_squared is not None:
        XX = check_array(X_norm_squared)
        if XX.shape == (1, X.shape[0]):
            XX = XX.T
        elif XX.shape != (X.shape[0], 1):
            raise ValueError(
                "Incompatible dimensions for X and X_norm_squared")
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]

    if X is Y:  # shortcut in the common case euclidean_distances(X, X)
        YY = XX.T
    elif Y_norm_squared is not None:
        YY = np.atleast_2d(Y_norm_squared)

        if YY.shape != (1, Y.shape[0]):
            raise ValueError(
                "Incompatible dimensions for Y and Y_norm_squared")
    else:
        YY = row_norms(Y, squared=True)[np.newaxis, :]

    distances = safe_sparse_dot(X, Y.T, dense_output=True)
    distances *= -2
    distances += XX
    distances += YY
    np.maximum(distances, 0, out=distances)

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances if squared else np.sqrt(distances, out=distances)

class TimeSeriesKMeans(TransformerMixin, ClusterMixin,
                       TimeSeriesCentroidBasedClusteringMixin,
                       BaseModelPackage, TimeSeriesBaseEstimator):

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-6, n_init=1,
                 metric="euclidean", max_iter_barycenter=100,
                 metric_params=None, n_jobs=None, dtw_inertia=False,
                 verbose=0, random_state=None, init='k-means++'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.metric = metric
        self.max_iter_barycenter = max_iter_barycenter
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.dtw_inertia = dtw_inertia
        self.verbose = verbose
        self.random_state = random_state
        self.init = init

    def _is_fitted(self):
        check_is_fitted(self, ['cluster_centers_'])
        return True

    def _get_metric_params(self):
        if self.metric_params is None:
            metric_params = {}
        else:
            metric_params = self.metric_params.copy()
        if "n_jobs" in metric_params.keys():
            del metric_params["n_jobs"]
        return metric_params

    def _fit_one_init(self, X, x_squared_norms, rs):
        metric_params = self._get_metric_params()
        n_ts, sz, d = X.shape
        if hasattr(self.init, '__array__'):
            self.cluster_centers_ = self.init.copy()
        elif isinstance(self.init, str) and self.init == "k-means++":
            if self.metric == "euclidean":
                self.cluster_centers_ = _kmeans_plusplus(
                    X.reshape((n_ts, -1)),
                    self.n_clusters,
                    x_squared_norms=x_squared_norms,
                    random_state=rs
                )[0].reshape((-1, sz, d))
            else:
                raise ValueError(
                        "Incorrect metric: %s (should be 'euclidean')" % self.metric
                    )
                self.cluster_centers_ = _k_init_metric(X, self.n_clusters,
                                                       cdist_metric=metric_fun,
                                                       random_state=rs)
        elif self.init == "random":
            indices = rs.choice(X.shape[0], self.n_clusters)
            self.cluster_centers_ = X[indices].copy()
        else:
            raise ValueError("Value %r for parameter 'init'"
                             "is invalid" % self.init)
        self.cluster_centers_ = _check_full_length(self.cluster_centers_)
        old_inertia = np.inf

        for it in range(self.max_iter):
            self._assign(X)
            if self.verbose:
                print("%.3f" % self.inertia_, end=" --> ")
            self._update_centroids(X)

            if np.abs(old_inertia - self.inertia_) < self.tol:
                break
            old_inertia = self.inertia_
        if self.verbose:
            print("")

        self._iter = it + 1

        return self

    def _transform(self, X):
        metric_params = self._get_metric_params()
        if self.metric == "euclidean":
            return cdist(X.reshape((X.shape[0], -1)),
                         self.cluster_centers_.reshape((self.n_clusters, -1)),
                         metric="euclidean")
        else:
            raise ValueError("Incorrect metric: %s (should be 'euclidean')" % self.metric)

    def _assign(self, X, update_class_attributes=True):
        dists = self._transform(X)
        matched_labels = dists.argmin(axis=1)
        if update_class_attributes:
            self.labels_ = matched_labels
            _check_no_empty_cluster(self.labels_, self.n_clusters)
            if self.dtw_inertia and self.metric != "dtw":
                inertia_dists = cdist_dtw(X, self.cluster_centers_,
                                          n_jobs=self.n_jobs,
                                          verbose=self.verbose)
            else:
                inertia_dists = dists
            self.inertia_ = _compute_inertia(inertia_dists,
                                             self.labels_,
                                             self._squared_inertia)
        return matched_labels

    def _update_centroids(self, X):
        metric_params = self._get_metric_params()
        for k in range(self.n_clusters):
                self.cluster_centers_[k] = euclidean_barycenter(
                    X=X[self.labels_ == k])

    def fit(self, X, y=None):

        X = check_array(X, allow_nd=True, force_all_finite='allow-nan')

        if hasattr(self.init, '__array__'):
            X = check_dims(X, X_fit_dims=self.init.shape,
                           extend=True,
                           check_n_features_only=(self.metric != "euclidean"))

        self.labels_ = None
        self.inertia_ = np.inf
        self.cluster_centers_ = None
        self._X_fit = None
        self._squared_inertia = True

        self.n_iter_ = 0

        max_attempts = max(self.n_init, 10)

        X_ = to_time_series_dataset(X)
        rs = check_random_state(self.random_state)

        if isinstance(self.init, str) and self.init == "k-means++" and \
                        self.metric == "euclidean":
            n_ts, sz, d = X_.shape
            x_squared_norms = cdist(X_.reshape((n_ts, -1)),
                                    np.zeros((1, sz * d)),
                                    metric="sqeuclidean").reshape((1, -1))
        else:
            x_squared_norms = None
        _check_initial_guess(self.init, self.n_clusters)

        best_correct_centroids = None
        min_inertia = np.inf
        n_successful = 0
        n_attempts = 0
        while n_successful < self.n_init and n_attempts < max_attempts:
            try:
                if self.verbose and self.n_init > 1:
                    print("Init %d" % (n_successful + 1))
                n_attempts += 1
                self._fit_one_init(X_, x_squared_norms, rs)
                if self.inertia_ < min_inertia:
                    best_correct_centroids = self.cluster_centers_.copy()
                    min_inertia = self.inertia_
                    self.n_iter_ = self._iter
                n_successful += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")
        self._post_fit(X_, best_correct_centroids, min_inertia)
        return self

    def fit_predict(self, X, y=None):

        X = check_array(X, allow_nd=True, force_all_finite='allow-nan')
        return self.fit(X, y).labels_

    def predict(self, X):

        X = check_array(X, allow_nd=True, force_all_finite='allow-nan')
        check_is_fitted(self, 'cluster_centers_')
        X = check_dims(X, X_fit_dims=self.cluster_centers_.shape,
                       extend=True,
                       check_n_features_only=(self.metric != "euclidean"))
        return self._assign(X, update_class_attributes=False)

    def transform(self, X):

        X = check_array(X, allow_nd=True, force_all_finite='allow-nan')
        check_is_fitted(self, 'cluster_centers_')
        X = check_dims(X, X_fit_dims=self.cluster_centers_.shape,
                       extend=True,
                       check_n_features_only=(self.metric != "euclidean"))
        return self._transform(X)

    def _more_tags(self):
        return {'allow_nan': True, 'allow_variable_length': True}


def initial_clusters(df):
    scaler = StandardScaler()
    tickers_scaled = scaler.fit_transform(df.iloc[:, 5:-1].T).T

    fig = px.line(tickers_scaled.T)#, y=tickers_scaled)
    fig.show()

    distortions = []
    silhouette = []
    K = range(1, 10)
    for k in tqdm(K):
        kmeanModel = TimeSeriesKMeans(n_clusters=k, metric="euclidean", n_jobs=6, max_iter=10)
        kmeanModel.fit(tickers_scaled)
        distortions.append(kmeanModel.inertia_)
        if k > 1:
            silhouette.append(silhouette_score(tickers_scaled, kmeanModel.labels_))

    distortions = pd.DataFrame(distortions, columns=['Distortion'])
    silhouette = pd.DataFrame(silhouette, columns=['Silhouette score'])

    fig = px.line(distortions, title='Elbow Method')
    fig.show()

    fig = px.line(silhouette, title='Silhouette')
    fig.show()
    
    return tickers_scaled

def final_clustering(df, n_clusters, tickers_scaled):
    ts_kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", n_jobs=3, max_iter=10)
    ts_kmeans.fit(tickers_scaled)

    plt.figure(figsize=(12,8))
    fig = px.line(ts_kmeans.cluster_centers_[range(0,n_clusters), :, 0].T)
    fig.show()

    df['cluster'] = ts_kmeans.predict(tickers_scaled)
    #pd.DataFrame(df.groupby('cluster')['sector'].value_counts())
    pd.DataFrame(df.groupby('cluster')['Industry'].value_counts())

    def plot_cluster_tickers(current_cluster):
        fig, ax = plt.subplots(
            int(np.ceil(current_cluster.shape[0]/4)),
            4,
            figsize=(15, 3*int(np.ceil(current_cluster.shape[0]/4)))
        )
        fig.autofmt_xdate(rotation=45)
        ax = ax.reshape(-1)

        for index, (_, row) in enumerate(current_cluster.iterrows()):
            ax[index].xaxis.set_ticks(np.arange(1, len(df.columns), len(df.columns)/5))
            ax.imshow
            ax[index].plot(row.iloc[2:-1])
            #ax[index].set_title(f"{row.shortName}\n{row.sector}")
            ax[index].set_title(f"{row.Ticker}\n{row.Industry}")
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    for cluster in range(n_clusters):
        print("=================================================================================")
        print(f"                          Cluster number: {cluster}")
        print("=================================================================================")
        plot_cluster_tickers(df[df.cluster==cluster])

def cointegration_test(df):
    col = []

    for i in tqdm(range(0, len(df.columns))):
        col1 = []
        for k in range(0, len(df.columns)):
            if i > k or i < k:
                col1.append(ts.coint(df.T.iloc[i], df.T.iloc[k])[1])
            else:
                col1.append(0)
        col.append(col1)
    return col

def plot_cointegration(df):
    df = -df
    fig, ax = plt.subplots()

    intersection_matrix = df
    xaxis = np.arange(len(df.index))
    ax.matshow(intersection_matrix, cmap='Greens', vmin=-0.051, vmax=0)
    ax.set_xticks(xaxis)
    ax.set_yticks(xaxis)
    ax.set_xticklabels(['']+df.index)
    ax.set_yticklabels(['']+df.index)
    ax.tick_params(axis='both', which='major', labelsize=80)
    plt.rcParams["figure.figsize"] = (200,200)
    plt.show()
    
    
def test_adf_select_stationary(df):
    stationary=[]
    for column in df.columns:
        result = adfuller(df[column])
        if result[1] < 0.05:
            print(column)
        #    print('ADF Statistic: %f' % result[0])
            print('p-value: %f' % result[1])
    #    print('Critical Values:')
    #     for key, value in result[4].items():
    #         print('\t%s: %.3f' % (key, value))
            stationary.append(column)
    return stationary