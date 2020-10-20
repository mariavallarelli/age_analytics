import pandas as pd
from numpy import unicode
from sklearn.base import BaseEstimator, TransformerMixin


class Selector(BaseEstimator, TransformerMixin):
    """class to select text or numeric feature type"""
    def __init__(self, key, dt):
        self.key = key
        self.dt = dt

    def fit(self, x, y=None):
        # does nothing
        return self

    def transform(self, data_dict):

        print("Selecting", self.key)
        if self.dt == 'text':
            return data_dict.loc[:, self.key]
        elif self.dt == 'num2text':
            return data_dict.loc[:, self.key].astype(unicode)
        elif self.dt == 'date':
            return data_dict.loc[:, self.key].squeeze() - pd.Timestamp(1900, 1, 1).dt.days.to_frame()
        else:
            return data_dict.loc[:, [self.key]].astype(float)
