"""Fixed length transformer, pad or truncate panel to fixed length."""

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.utils.pandas import df_map

__all__ = ["FixedLengthTransformer"]
__author__ = ["user"]


class FixedLengthTransformer(BaseTransformer):
    """Transform panel of variable length time series to fixed length.

    Transforms input dataset to a fixed length by either:
    - Padding shorter series with a fill value (default: 0)
    - Truncating longer series to the specified length

    Unlike PaddingTransformer, this transformer requires a fixed_length parameter
    and will both pad and truncate as needed.

    Parameters
    ----------
    fixed_length : int
        The exact length that all series will be transformed to
    fill_value : any, optional (default=0)
        The value used to pad shorter series

    Example
    -------
    >>> import pandas as pd
    >>> from sktime.transformations.panel.fixed_length import FixedLengthTransformer
    >>>
    >>> # Create a sample nested DataFrame with unequal length time series
    >>> data = {
    ...     'feature1': [
    ...         pd.Series([1, 2, 3]), pd.Series([4, 5]), pd.Series([6, 7, 8, 9])
    ...     ],
    ...     'feature2': [
    ...         pd.Series([10, 11]), pd.Series([12, 13, 14]), pd.Series([15])
    ...     ]
    ... }
    >>> X = pd.DataFrame(data)
    >>>
    >>> # Initialize the FixedLengthTransformer with fixed_length=3
    >>> transformer = FixedLengthTransformer(fixed_length=3)
    >>>
    >>> # Fit the transformer to the data
    >>> transformer.fit(X)
    >>>
    >>> # Transform the data
    >>> Xt = transformer.transform(X)
    >>>
    >>> # Display the transformed data
    >>> print(Xt)
    """

    _tags = {
        "authors": ["user"],
        "maintainers": ["user"],
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": False,
        "X_inner_mtype": "nested_univ",
        "y_inner_mtype": "None",
        "fit_is_empty": True,  # No need to compute anything during fit
        "capability:unequal_length:removes": True,
    }

    def __init__(self, fixed_length, fill_value=0):
        if fixed_length is None or fixed_length <= 0:
            raise ValueError("fixed_length must be a positive integer")
            
        self.fixed_length = fixed_length
        self.fill_value = fill_value
        super().__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        This is a no-op since we only need the fixed_length parameter.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
        y : ignored argument for interface compatibility

        Returns
        -------
        self : reference to self
        """
        return self

    def _transform_series(self, series):
        """Transform a single series to fixed length by padding or truncating.

        Parameters
        ----------
        series : pandas.Series
            The input series to transform

        Returns
        -------
        numpy.ndarray
            Fixed length array
        """
        series_length = len(series)
        
        if series_length == self.fixed_length:
            # Series is already the correct length
            return series.values
        elif series_length < self.fixed_length:
            # Pad the series with fill_value
            result = np.full(self.fixed_length, self.fill_value, dtype=float)
            result[:series_length] = series.iloc[:series_length]
            return result
        else:
            # Truncate the series
            return series.iloc[:self.fixed_length].values

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of Xt contains pandas.Series with fixed length
        """
        n_instances, _ = X.shape

        # Process each row of instances
        transformed_rows = []
        for i in range(n_instances):
            # Transform each series in the row
            row_series = X.iloc[i, :].values
            transformed_series = [pd.Series(self._transform_series(series)) 
                                 for series in row_series]
            transformed_rows.append(pd.Series(transformed_series))
        
        # Convert back to DataFrame
        Xt = df_map(pd.DataFrame(transformed_rows))(pd.Series)
        Xt.columns = X.columns

        return Xt