"""Module comprising the bootstrapping algorithm for indicators."""

from __future__ import annotations
from inspect import signature
from math import ceil, floor
from typing import Any, Callable, Generator

import cftime
import numpy as np
import xarray
from boltons.funcutils import wraps
from xarray.core.dataarray import DataArray

import pandas as pd
import xclim.core.utils

from .calendar import convert_calendar, parse_offset

BOOTSTRAP_DIM = "_bootstrap"


def percentile_bootstrap(func):
    """Decorator applying a bootstrap step to the calculation of exceedance over a percentile threshold.

    This feature is experimental.

    Bootstrapping avoids discontinuities in the exceedance between the reference period over which percentiles are
    computed, and "out of reference" periods. See `bootstrap_func` for details.

    Declaration example:

    .. code-block:: python

        @declare_units(tas="[temperature]", t90="[temperature]")
        @percentile_bootstrap
        def tg90p(
            tas: xarray.DataArray,
            t90: xarray.DataArray,
            freq: str = "YS",
            bootstrap: bool = False,
        ) -> xarray.DataArray:
            pass

    Examples
    --------
    >>> from xclim.core.calendar import percentile_doy
    >>> from xclim.indices import tg90p
    >>> tas = xr.open_dataset(path_to_tas_file).tas
    >>> # To start bootstrap reference period must not fully overlap the studied period.
    >>> tas_ref = tas.sel(time=slice("1990-01-01", "1992-12-31"))
    >>> t90 = percentile_doy(tas_ref, window=5, per=90)
    >>> tg90p(tas=tas, tas_per=t90.sel(percentiles=90), freq="YS", bootstrap=True)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        ba = signature(func).bind(*args, **kwargs)
        ba.apply_defaults()
        bootstrap = ba.arguments.get("bootstrap", False)
        if bootstrap is False:
            return func(*args, **kwargs)

        return bootstrap_func(func, **ba.arguments)

    return wrapper


def bootstrap_func(
    compute_index_func: Callable[[], DataArray], **index_kwargs
) -> DataArray:
    r"""Bootstrap the computation of percentile-based indices.

    Indices measuring exceedance over percentile-based thresholds (such as tx90p) may contain artificial discontinuities
    at the beginning and end of the reference period used to calculate percentiles. The bootstrap procedure can reduce
    those discontinuities by iteratively computing the percentile estimate and the index on altered reference periods.

    These altered reference periods are themselves built iteratively: When computing the index for year `x`, the
    bootstrapping creates as many altered reference periods as the number of years in the reference period.
    To build one altered reference period, the values of year `x` are replaced by the values of another year in the
    reference period, then the index is computed on this altered period. This is repeated for each year of the reference
    period, excluding year `x`. The final result of the index for year `x` is then the average of all the index results
    on altered years.

    Parameters
    ----------
    compute_index_func : Callable
        Index function.
    \*\*kwargs
        Arguments to `func`.

    Returns
    -------
    xr.DataArray
        The result of func with bootstrapping.

    References
    ----------
    :cite:cts:`zhang_avoiding_2005`

    Notes
    -----
    This function is meant to be used by the `percentile_bootstrap` decorator.
    The parameters of the percentile calculation (percentile, window, reference_period)
    are stored in the attributes of the percentile DataArray.
    The bootstrap algorithm implemented here does the following::

        For each temporal grouping in the calculation of the index
            If the group `g_t` is in the reference period
                For every other group `g_s` in the reference period
                    Replace group `g_t` by `g_s`
                    Compute percentile on resampled time series
                    Compute index function using percentile
                Average output from index function over all resampled time series
            Else compute index function using original percentile

    """
    # Identify the input and the percentile arrays from the bound arguments
    per_key = None
    da_key = None
    for name, val in index_kwargs.items():
        if isinstance(val, DataArray):
            if "percentile_doy" in val.attrs.get("history", ""):
                per_key = name
            else:
                da_key = name
    if da_key is None or per_key is None:
        msg = "bootstrap decorator can only be applied to indices with 2 Dataarrays."
        raise NotImplementedError(msg)
    # Extract the DataArray inputs from the arguments
    da: DataArray = index_kwargs.pop(da_key)
    per_da: DataArray | None = index_kwargs.pop(per_key, None)
    if per_da is None:
        # per may be empty on non doy percentiles
        raise KeyError(
            "`bootstrap` can only be used with percentiles computed using `percentile_doy`"
        )
    # Boundary years of reference period
    clim = per_da.attrs["climatology_bounds"]
    # overlap of studied `da` and reference period used to compute percentile
    overlap_da = da.sel(time=slice(*clim))
    if len(overlap_da.time) == len(da.time):
        raise KeyError(
            "`bootstrap` is unnecessary when all years are overlapping between reference "
            "(percentiles period) and studied (index period) periods."
        )
    if len(overlap_da) == 0:
        raise KeyError(
            "`bootstrap` is unnecessary when no year are overlapping between the reference "
            "(percentiles period) and the studied (index period) periods."
        )
    per_window = per_da.attrs["window"]
    per_alpha = per_da.attrs["alpha"]
    per_beta = per_da.attrs["beta"]
    per_per = per_da.percentiles.data[()]
    bfreq = _get_bootstrap_freq(index_kwargs["freq"])
    # Group input array in years, with an offset matching freq
    # TODO: Use `overlap_years_groups` instead of `in_base_years` to generalize for freq != YS
    overlap_years_groups = overlap_da.resample(time=bfreq).groups
    da_years_groups = da.resample(time=bfreq).groups
    per_template = per_da.copy(deep=True)
    exceedances = []
    # Compute bootstrapped index on each year of overlapping years

    ##################
    #  Attempt start #
    #    One sort    #
    ##################
    # -- Compute index before in_base
    in_base_years = [_get_year_label(y) for y in overlap_years_groups.keys() ]
    # -- Compute index before in_base
    inbase_first_year = min(in_base_years)
    before_inbase_first_year = da.get_index("time").year[-1]
    if before_inbase_first_year < inbase_first_year  :
        # -- There are years before in_base, compute the index normally on them
        before_in_base_slice = slice(before_inbase_first_year, inbase_first_year -1)
        index_kwargs[da_key] = da.sel(time=before_in_base_slice)
        index_kwargs[per_key] = per_da
        before_inbase_exceedance = compute_index_func(**index_kwargs)
        exceedances.append(before_inbase_exceedance)

    crd = xarray.Coordinates.from_pandas_multiindex(
        pd.MultiIndex.from_arrays(
            (overlap_da.time.dt.year.values, overlap_da.time.dt.dayofyear.values),
            names=("year", "dayofyear"),
        ),
        "time",
    )
    windowed_data = (
        overlap_da
        .assign_coords(crd)
        .unstack("time")
        .rolling(min_periods=1, center=True, dayofyear=per_window)
        .construct("window")
    )
    index_kwargs[da_key] = overlap_da
    # TODO: make sure it can fit in memory or that dask is clever enough to not persist it if it does n.
    windowed_data.persist()  # persist `windowed_data` as we reuse it many times
    sorted_stacked_wo_replaced_year = None
    # TODO: use overlap_years_groups instead, in case the freq is 10YS or YS-JUNE
    for year_to_be_replaced in in_base_years:
        # loop over the "30" years of in_base
        # pop year_to_be_replaced
        windowed_data_without_year_to_be_replaced = windowed_data.where(
            windowed_data.time.dt.year == year_to_be_replaced, drop=True
        )
        # replaced_year = data.pop(year_to_be_replaced)
        stacked_wo_replaced_year = windowed_data_without_year_to_be_replaced.stack(stacked_dim=("window", "year"))
        # `stacked` dim is at the end of of stacked_data (which is actually better for sort)
        # TODO: use map_block to avoid putting the whole array in memory with sort ?
        #       (perhaps it's actually better to have it in mem, if it fits)
        # TODO 2: See if we can use topk instead of sort.
        #         This might reduce greatly the memory cost.
        #         In which case, we would need to rewrite calc_perc to
        #         select the left and right bounds from the subsample (the topk appended to the replacing_year)
        #         and compute the interpolation ourselves.
        sorted_stacked_wo_replaced_year = np.sort(stacked_wo_replaced_year.data, axis=-1)
        summed_exceedance: DataArray | None = None
        replacing_years = (windowed_data_without_year_to_be_replaced.sel(year=y).stack(stacked=("window", "year")).data
                           for y in in_base_years
                           if y != year_to_be_replaced)
        for replacing_year_data in replacing_years:
            # loop over the "29" other years and compute exceedance
            sorted_stacked_rebuilt_in_base = np.sort(
                np.append(sorted_stacked_wo_replaced_year, replacing_year_data),
                axis=-1,
                kind="mergesort",
            )
            per_doy = xclim.core.utils.calc_perc(
                sorted_stacked_rebuilt_in_base,
                alpha=per_alpha,
                beta=per_beta,
                percentiles=per_per,
                copy=False,
                pre_sorted=True
            )
            index_kwargs[per_key] = per_doy
            if summed_exceedance is None:
                summed_exceedance = compute_index_func(**index_kwargs)
            else:
                summed_exceedance += compute_index_func(**index_kwargs)
        if summed_exceedance is None:
            err = "No year were bootstrapped."
            raise NotImplementedError(err)
        averaged_in_base_exceedance = summed_exceedance / (len(in_base_years) - 1)
        exceedances.append(averaged_in_base_exceedance)
    # -- Compute index after in_base
    in_base_last_year = np.max(in_base_years)
    after_base_last_year = da.time.isel(time=-1).time.dt.year
    if after_base_last_year > in_base_last_year :
        # -- There are years after in_base, compute the index normally on them
        after_in_base_slice = slice(in_base_last_year + 1, after_base_last_year)
        index_kwargs[per_key] = per_da
        index_kwargs[da_key] = da.sel(time=after_in_base_slice)
        after_inbase_exceedance = compute_index_func(**index_kwargs)
        exceedances.append(after_inbase_exceedance)
    # -- Aggregate results
    result = xarray.concat(exceedances, dim="time")
    result.attrs["units"] = exceedances[0].attrs["units"]

    ##################
    #   Attempt end  #
    ##################

    # value = []
    # for year_key, year_slice in da_years_groups.items():
    #     # 30 years
    #     kw = {da_key: da.isel(time=year_slice), **index_kwargs}
    #     if _get_year_label(year_key) in overlap_da.get_index("time").year:
    #         # If the group year is in both reference and studied periods, run the bootstrap
    #         exceedances = []
    #
    #         # TODO: optimize here ?
    #         # Cherry-picking from percentile_doy, we could pre-build `rolled_overlap_da` (lat, lon, doy, window_year)
    #         # dataset.
    #         # Then we create `sorted_overlap_da` by sorting on window_year
    #         # Then we build `rolled_bda`: the (lat, lon, doy, window) of the year to add to the dataset (year_slice).
    #         # Then we only need to concat the `rolled_bda` with `sorted_overlap_da` (would replace `build_bootstrap_year_da`).
    #         # run cal_perc on `concated_da` with apply_ufunc.
    #
    #         for bda_year in build_bootstrap_year_da(
    #             overlap_da, overlap_years_groups, year_key
    #         ):
    #             # 30 years
    #             per = percentile_doy(
    #                 bda_year,
    #                 window=per_window,
    #                 alpha=per_alpha,
    #                 beta=per_beta,
    #                 per=per_per,
    #             )
    #             kw[per_key] = per
    #             exceedances.append(compute_index_func(**kw))
    #         value = xarray.concat(exceedances, dim=BOOTSTRAP_DIM).mean(
    #             dim=BOOTSTRAP_DIM, keep_attrs=True
    #         )
    #     else:
    #         # Otherwise, run the normal computation using the original percentile
    #         kw[per_key] = per_da
    #         value = compute_index_func(**kw)
    #     value.append(value)
    # result = xarray.concat(value, dim="time")
    # result.attrs["units"] = value.attrs["units"]
    return result

from  dask.array.core import Array
def distributed_percentile(arr: Array, per: int, axis:int):
    """
    Parameters
    ----------
    arr: dask.Array
    per: int
        Percentile value. Must be between 0 and 99.
    axis: int
        Axis where the computation is performed.
    """
    import dask
    # Monkey patch dask chunk's topk to use our nan handling topk
    dask.array.chunk.topk = nan_topk
    quantile = per/100
    # Ideal because some chunks may have Nan, so the number of values below per will be lower
    ideal_number_of_value_below_per = arr.size * quantile
    ideal_number_of_value_below_per_ceil = ceil(ideal_number_of_value_below_per)
    ideal_number_of_value_below_per_floor = floor(ideal_number_of_value_below_per)
    top_per_percent = arr.topk(k = ideal_number_of_value_below_per_ceil, axis=axis)
    if has_no_nans(arr):
        if ideal_number_of_value_below_per_ceil == ideal_number_of_value_below_per_floor:
            # No need to interpolate, return smallest value of the top per percent values
            return top_per_percent[-1]
        else:
            result = linear_interpolation(top_per_percent[-1], top_per_percent[-2])
    else:
        # has nans
        sample_sizes = get_no_nans_sample_sizes()
        virtual_indexes = _compute_virtual_index(n = sample_sizes, quantiles=quantile, alpha=1/3, beta=1/3)
        previous_indexes, next_indexes = _get_indexes(
        arr, virtual_indexes, valid_values_count=ideal_number_of_value_below_per_ceil
        )
        # Same shape array but for the first axis which is one value (will be filled with the percentile)
        result = arr[1,...]
        # TODO: make sure this is performant enough
        #       There is no take_along_axis on dask so we need an alternative
        previouses = top_per_percent.ravel()[previous_indexes.ravel()].reshape(previous_indexes)
        nextes = top_per_percent.ravel()[next_indexes.ravel()].reshape(next_indexes)

        for i, (b,a) in enumerate(zip(previous_indexes, next_indexes)):
            result[i] = linear_interpolation(top_per_percent[i,-b], top_per_percent[i,-a])



def _compute_virtual_index(
    n: np.ndarray, quantiles: np.ndarray | float, alpha: float, beta: float
):
    """Compute the floating point indexes of an array for the linear interpolation of quantiles.

    Based on the approach used by :cite:t:`hyndman_sample_1996`.

    Parameters
    ----------
    n : array_like
        The sample sizes.
    quantiles : array_like | float
        The quantiles values.
    alpha : float
        A constant used to correct the index computed.
    beta : float
        A constant used to correct the index computed.

    Notes
    -----
    `alpha` and `beta` values depend on the chosen method (see quantile documentation).

    References
    ----------
    :cite:cts:`hyndman_sample_1996`
    """
    return n * quantiles + (alpha + quantiles * (1 - alpha - beta)) - 1

def linear_interpolation(arr1, arr2, alpha: float = 1.0, beta: float = 1.0):
    pass


def nan_topk(a, k, axis, keepdims):
    """Compute topk on a chunk while ignoring nans.

    TODO: To upstream to Dask
    """
    assert keepdims is True
    axis = axis[0]
    if abs(k) >= a.shape[axis]:
        return a
    max_nan_count = np.isnan(a).sum(axis=axis).max()
    if max_nan_count != 0:
        a = np.partition(a, -(k + max_nan_count), axis=axis)
        a = a[~np.isnan(a)]
    else:
        a = np.partition(a, -k, axis=axis)
    k_slice = slice(-k, None) if k > 0 else slice(-k)
    return a[tuple(k_slice if i == axis else slice(None) for i in range(a.ndim))]

def nan_argtopk(a_plus_idx, k, axis, keepdims):
    """Compute argtopk on a chunk while ignoring nans.

    TODO: To upstream to Dask
    """
    from dask.core import flatten
    assert keepdims is True
    axis = axis[0]
    if isinstance(a_plus_idx, list):
        a_plus_idx = list(flatten(a_plus_idx))
        a = np.concatenate([ai for ai, _ in a_plus_idx], axis)
        idx = np.concatenate(
            [np.broadcast_to(idxi, ai.shape) for ai, idxi in a_plus_idx], axis
        )
    else:
        a, idx = a_plus_idx
    if abs(k) >= a.shape[axis]:
        return a_plus_idx
    max_nan_count = np.isnan(a).sum(axis=axis).max()
    if max_nan_count != 0:
        idx2 = np.argpartition(a, -(k + max_nan_count), axis=axis)
        mask = np.isnan(a[idx2])
        idx2 = idx2[~mask]
    else:
        idx2 = np.argpartition(a, -k, axis=axis)
    k_slice = slice(-k, None) if k > 0 else slice(-k)
    idx2 = idx2[tuple(k_slice if i == axis else slice(None) for i in range(a.ndim))]
    return np.take_along_axis(a, idx2, axis), np.take_along_axis(idx, idx2, axis)


def _get_bootstrap_freq(freq):
    _, base, start_anchor, anchor = parse_offset(freq)  # noqa
    bfreq = "Y"
    if start_anchor:
        bfreq += "S"
    else:
        bfreq += "E"
    if base in ["A", "Y", "Q"] and anchor is not None:
        bfreq = f"{bfreq}-{anchor}"
    return bfreq


def _get_year_label(year_dt: cftime.datetime | np.datetime64) -> str:
    if isinstance(year_dt, cftime.datetime):
        year_label = year_dt.year
    else:
        year_label = year_dt.astype("datetime64[Y]").astype(int) + 1970
    return year_label


# TODO: Return a generator instead and assess performance
def build_bootstrap_year_da(
    da: DataArray, groups: dict[Any, slice], label: Any, dim: str = "time"
) -> Generator[DataArray]:
    """Return an array where a group in the original is replaced by every other groups along a new dimension.

    Parameters
    ----------
    da : DataArray
      Original input array over reference period.
    groups : dict
      Output of grouping functions, such as `DataArrayResample.groups`.
    label : Any
      Key identifying the group item to replace.
    dim : str
      Dimension recognized as time. Default: `time`.

    Returns
    -------
    Generator[DataArray]:
      Array where one group is replaced by values from every other group along the `bootstrap` dimension.
    """
    gr = groups.copy()
    # Location along dim that must be replaced
    bloc = da[dim][gr.pop(label)]
    # Initialize output array with new bootstrap dimension
    out = da.expand_dims({BOOTSTRAP_DIM: np.arange(len(gr))}).copy(deep=True)
    # Replace `bloc` by every other group
    for i, (_, group_slice) in enumerate(gr.items()):
        source = da.isel({dim: group_slice})
        out_view = out.loc[{BOOTSTRAP_DIM: i}]
        if len(source[dim]) < 360 and len(source[dim]) < len(bloc):
            # This happens when the sampling frequency is anchored thus
            # source[dim] would be only a few months on the first and last year
            pass
        elif len(source[dim]) == len(bloc):
            out_view.loc[{dim: bloc}] = source.data
        elif len(bloc) == 365:
            out_view.loc[{dim: bloc}] = convert_calendar(source, "365_day").data
        elif len(bloc) == 366:
            out_view.loc[{dim: bloc}] = convert_calendar(
                source, "366_day", missing=np.NAN
            ).data
        elif len(bloc) < 365:
            # 360 days calendar case or anchored years for both source[dim] and bloc case
            out_view.loc[{dim: bloc}] = source.data[: len(bloc)]
        else:
            raise NotImplementedError
        yield out_view
