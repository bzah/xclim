from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xclim.indices.fire import (
    griffiths_drought_factor,
    keetch_byram_drought_index,
    mcarthur_forest_fire_danger_index,
)


@pytest.mark.parametrize(
    "p,t,pa,k0,exp",
    [
        (10 * [100], 10 * [0], [1.0], [0.0], 0.0),
        (10 * [0], 10 * [100], [1.0], [0.0], 203.2),
        ([10, 0, 0.1, 6, 0, 0, 0.5, 0.3, 0, 1], 10 * [30], [1.0], [0.0], 7.25278),
        (10 * [0], [20, 30, 20, 30, 30, 25, 40, 35, 20, 20], [1.0], [0.0], 8.46632),
        (
            [10, 0, 0.1, 6, 0, 0, 0.5, 0.3, 0, 1],
            [20, 30, 20, 30, 30, 25, 40, 35, 20, 20],
            [1.0],
            [0.0],
            7.10174,
        ),
        (
            [10, 0, 0.1, 6, 0, 0, 0.5, 0.3, 0, 1],
            [20, 30, 20, 30, 30, 25, 40, 35, 20, 20],
            [1.0],
            [10.0],
            12.18341,
        ),
        (
            [10, 0, 0.1, 6, 0, 0, 0.5, 0.3, 0, 1],
            [20, 30, 20, 30, 30, 25, 40, 35, 20, 20],
            [100.0],
            [0.0],
            8.45569,
        ),
        (
            [10, 0, 0.1, 6, 0, 0, 0.5, 0.3, 0, 1],
            [20, 30, 20, 30, 30, 25, 40, 35, 20, 20],
            [1.0],
            [203.2],
            197.33375,
        ),
    ],
)
def test_keetch_byram_drought_index_2(p, t, pa, k0, exp, pr_series, tasmax_series):
    """Compare output to calculation by hand"""
    pr = pr_series(p, units="mm/day")
    tasmax = tasmax_series(t, units="degC")
    pr_annual = xr.DataArray(pa, attrs={"units": "mm/year"})
    kbdi0 = xr.DataArray(k0, attrs={"units": "mm/day"})

    kbdi_final = keetch_byram_drought_index(pr, tasmax, pr_annual, kbdi0).isel(time=-1)
    np.testing.assert_allclose(kbdi_final, exp, atol=1e-5)


@pytest.mark.parametrize(
    "p, s, exp, test_discrete",
    [
        (17 * [0] + [5, 10, 20], 20 * [10], 0.40471, False),
        ([20, 10, 5] + 17 * [0], 20 * [10], 6.13148, True),
        (
            [0, 30, 5, 0, 0, 5, 10, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3, 1],
            20 * [30],
            6.82454,
            True,
        ),
        (
            [0, 10, 5, 0, 0, 5, 10, 0, 0, 20, 0, 0, 0, 20, 0, 0, 0, 5, 4, 3],
            20 * [30],
            6.59186,
            False,
        ),
        (
            [0, 10, 5, 0, 0, 50, 100, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 1, 3, 1],
            20 * [10],
            3.91578,
            False,
        ),
        (
            [0, 300, 5, 0, 0, 50, 100, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 1, 3, 1],
            20 * [30],
            3.76635,
            False,
        ),
    ],
)
def test_griffiths_drought_factor(p, exp, s, test_discrete, pr_series):
    """Compare output for a single window to calculation by hand"""
    pr = pr_series(p, units="mm/day")
    smd = pr_series(s, units="mm/day")

    df = griffiths_drought_factor(pr, smd, "xlim")
    np.testing.assert_allclose(df, exp, atol=1e-5)

    if test_discrete:
        df = griffiths_drought_factor(pr, smd, "discrete")
        np.testing.assert_allclose(df, round(exp), atol=1e-5)


def test_griffiths_drought_factor_sliding(pr_series):
    """Compare output for a simple case to calculation by hand"""
    p = np.zeros(24)
    p[19] = 20.0
    pr = pr_series(p, units="mm/day")
    smd = pr_series(20 * np.ones(24), units="mm/day")
    exp = np.array([1.07024, 3.14744, 4.71645, 5.64112, 6.14665])

    df = griffiths_drought_factor(pr, smd, "xlim")
    np.testing.assert_allclose(df, exp, atol=1e-5)


def test_mcarthur_forest_fire_danger_index(
    pr_series, tasmax_series, hurs_series, sfcWind_series
):
    """Compare output to calculation by hand"""
    D = pr_series(range(1, 11), units="")  # This is probably not good practice?
    T = tasmax_series(range(30, 40), units="degC")
    H = hurs_series(range(10, 20))
    V = sfcWind_series(range(10, 20))

    # Compare FFDI to values calculated using original arrangement of the FFDI:
    exp = 2 * np.exp(-0.450 + 0.987 * np.log(D) - 0.0345 * H + 0.0338 * T + 0.0234 * V)
    ffdi = mcarthur_forest_fire_danger_index(D, T, H, V)
    np.testing.assert_allclose(ffdi, exp, rtol=1e-6)
