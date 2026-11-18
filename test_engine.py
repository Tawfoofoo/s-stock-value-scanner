"""Offline regression tests for engine.py. Run: python -m pytest -q test_engine.py"""
import math
import numpy as np
import pandas as pd
import pytest
import sys
import types

# Offline tests can run without Streamlit/yfinance installed.
try:
    import streamlit
except ImportError:
    stub = types.ModuleType('streamlit')
    stub.cache_data = lambda **kwargs: (lambda func: func)
    sys.modules['streamlit'] = stub
try:
    import yfinance
except ImportError:
    sys.modules['yfinance'] = types.ModuleType('yfinance')
import engine


def test_dcf_known_constant_cash_flow():
    # A perpetuity of 100 with 0% growth and 10% discount rate is worth 1000.
    assert engine.dcf(100, 0, 0, 10, 0, 0, .10) == pytest.approx(100)


def test_dcf_no_double_count_cash_debt():
    a = engine.dcf(100, 0, 0, 10, .03, .02, .10)
    b = engine.dcf(100, 10000, 5000, 10, .03, .02, .10)
    assert a == pytest.approx(b)


@pytest.mark.parametrize('kwargs', [
    dict(fcf=-1), dict(shares=0), dict(rate=.02, terminal=.03),
    dict(rate=-1.1, terminal=-1.2), dict(growth=-1.1), dict(fcf=float('nan')),
])
def test_dcf_invalid(kwargs):
    inputs = dict(fcf=100, cash=0, debt=0, shares=10, growth=.03, terminal=.02, rate=.10)
    inputs.update(kwargs)
    assert engine.dcf(**inputs) is None


def test_quality_negative_debt_to_equity_not_rewarded():
    baseline = engine.quality_score(.2, None, [4, 3, 2, 1], [4, 3, 2], [4, 3, 2], 1)
    assert engine.quality_score(.2, -.5, [4, 3, 2, 1], [4, 3, 2], [4, 3, 2], 1) == baseline
    assert engine.quality_score(.2, .2, [4, 3, 2, 1], [4, 3, 2], [4, 3, 2], 1) == baseline + 20


def test_annual_series_stops_at_gap():
    d = {'2025-12-31': 100, '2024-12-31': 90, '2022-12-31': 70, '2021-12-31': 60}
    assert len(engine.annual_series(d)) == 2


def test_annual_series_consecutive():
    d = {'2025-12-31': 100, '2024-12-31': 90, '2023-12-31': 80, '2022-12-31': 70}
    assert len(engine.annual_series(d)) == 4


def fake_prices():
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=260)
    close = pd.Series(np.linspace(80, 100, 260), index=idx)
    return pd.DataFrame({'Open': close, 'High': close + 1, 'Low': close - 1,
                         'Close': close, 'Volume': 1_000_000}, index=idx)


def fake_statements():
    years = [pd.Timestamp(f'{y}-12-31') for y in (2025, 2024, 2023, 2022)]
    cf = pd.DataFrame({dt: {'Free Cash Flow': 2e9 - i * 1e8,
                            'Operating Cash Flow': 3e9 - i * 1e8}
                       for i, dt in enumerate(years)})
    bs = pd.DataFrame({years[0]: {'Stockholders Equity': 10e9, 'Total Debt': 2e9}})
    inc = pd.DataFrame({dt: {'Total Revenue': 20e9 - i * 1e9,
                             'Net Income': 1e9 - i * 1e8}
                        for i, dt in enumerate(years)})
    info = {'shortName': 'Test Co', 'sharesOutstanding': 1e9,
            'returnOnEquity': .2, 'debtToEquity': 20,
            'totalCash': 3e9, 'totalDebt': 2e9}
    return info, {}, cf, bs, inc


def test_evaluate_contract_without_network(monkeypatch):
    monkeypatch.setattr(engine, 'history', lambda symbol: fake_prices())
    monkeypatch.setattr(engine, 'fundamentals', lambda symbol: fake_statements())
    result, detail = engine.evaluate('TEST', .10, .10, .025, 20, 70, 1.0, 'Technology')
    expected = ['Ticker', 'Price ($)', 'Fair Price ($)', 'Bear FV ($)', 'Base FV ($)',
                'Bull FV ($)', 'Upside (%)', 'MOS (%)', 'Quality Score',
                'Valuation Confidence', 'CDC Zone', 'CDC Buy today', 'CDC Sell today',
                'Eligible', 'Decision', 'FCF years', 'Growth (%)']
    assert all(key in result for key in expected)
    assert result['Fair Price ($)'] == result['Base FV ($)']
    assert result['Bear FV ($)'] < result['Base FV ($)'] < result['Bull FV ($)']
    assert result['Upside (%)'] == pytest.approx((result['Fair Price ($)'] / result['Price ($)'] - 1) * 100)
    assert result['Eligible'] == (result['Upside (%)'] >= 20 and result['Quality Score'] >= 70 and result['Valuation Confidence'] != 'Low')
    assert len(detail) == 260


def test_special_sector_excluded():
    with pytest.raises(ValueError, match='Special Review'):
        engine.evaluate('BANK', sector='Financials')
