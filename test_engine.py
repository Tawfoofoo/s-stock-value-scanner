"""Offline regression tests: python -m unittest -v test_engine.py"""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import sys
import types
try:
    import streamlit
except ImportError:
    st = types.ModuleType('streamlit')
    st.cache_data = lambda **kwargs: (lambda f: f)
    sys.modules['streamlit'] = st
try:
    import yfinance
except ImportError:
    sys.modules['yfinance'] = types.ModuleType('yfinance')
import engine

DATES = pd.to_datetime(['2025-12-31', '2024-12-31', '2023-12-31', '2022-12-31'])


def statements():
    cf = pd.DataFrame({d: {'Free Cash Flow': v, 'Operating Cash Flow': op, 'Capital Expenditure': cap}
                       for d, v, op, cap in zip(DATES, [100., np.nan, 80., 70.],
                                                 [140., 130., 120., 110.], [-40., -40., -40., -40.])})
    inc = pd.DataFrame({d: {'Total Revenue': 300. + i * 10, 'Net Income': 60. + i * 5}
                        for i, d in enumerate(DATES)})
    bs = pd.DataFrame({DATES[0]: {'Stockholders Equity': 1000., 'Total Debt': 100.}})
    return cf, bs, inc


class EngineRegression(unittest.TestCase):
    def test_direct_fcf_filled_per_year(self):
        cf, _, _ = statements()
        values = engine.annual_fcf_map(cf)
        self.assertEqual(values['2025-12-31'], 100.)
        self.assertEqual(values['2024-12-31'], 90.)
        self.assertEqual(values['2023-12-31'], 80.)
        self.assertEqual(len(engine.annual_series(values)), 4)

    def test_zero_fcf_is_not_replaced(self):
        cf, _, _ = statements()
        cf.loc['Free Cash Flow', DATES[0]] = 0
        self.assertEqual(engine.annual_fcf_map(cf)['2025-12-31'], 0)

    def test_alias_fills_missing_year(self):
        cf = pd.DataFrame({DATES[0]: {'Free Cash Flow': np.nan, 'FreeCashFlow': 12},
                           DATES[1]: {'Free Cash Flow': 11, 'FreeCashFlow': np.nan}})
        self.assertEqual(engine.dated_values(cf, ['Free Cash Flow', 'FreeCashFlow']),
                         {'2025-12-31': 12., '2024-12-31': 11.})

    def test_missing_middle_year_is_rejected(self):
        values = {'2025-12-31': 10, '2024-12-31': None, '2023-12-31': 9, '2022-12-31': 8}
        with self.assertRaisesRegex(ValueError, 'Insufficient consecutive'):
            engine.fcf_history_or_error(values)

    def test_missing_year_not_fabricated(self):
        values = {'2025-12-31': 10, '2023-12-31': 9, '2022-12-31': 8}
        with self.assertRaisesRegex(ValueError, 'Insufficient consecutive'):
            engine.fcf_history_or_error(values)

    def test_special_sector_still_skipped(self):
        with self.assertRaisesRegex(ValueError, 'Special Review'):
            engine.evaluate('JPM', sector='Financials')

    def test_dcf_no_cash_debt_double_count(self):
        self.assertAlmostEqual(engine.dcf(100, 0, 0, 10, .05, .025, .10),
                               engine.dcf(100, 1000, 100, 10, .05, .025, .10))

    def test_evaluate_contract(self):
        cf, bs, inc = statements()
        ix = pd.date_range('2025-01-01', periods=250, freq='B')
        hist = pd.DataFrame({'Open': 10., 'High': 11., 'Low': 9.,
                             'Close': np.linspace(10, 11, 250), 'Volume': 10000.}, index=ix)
        info = {'sharesOutstanding': 100., 'returnOnEquity': .22, 'debtToEquity': 10., 'shortName': 'Test'}
        with patch.object(engine, 'history', return_value=hist), patch.object(
                engine, 'fundamentals', return_value=(info, {}, cf, bs, inc)):
            row, technical_frame = engine.evaluate('TEST', .10, .10, .025, 20, 70, 1., 'Industrials')
        required = {'Ticker', 'Price ($)', 'Fair Price ($)', 'Bear FV ($)', 'Base FV ($)',
                    'Bull FV ($)', 'Upside (%)', 'MOS (%)', 'Quality Score', 'Eligible',
                    'Valuation Confidence', 'CDC Zone', 'CDC Buy today', 'Decision'}
        self.assertTrue(required.issubset(row))
        self.assertEqual(row['FCF years'], 4)
        self.assertEqual(row['Fair Price ($)'], row['Base FV ($)'])
        self.assertIsInstance(technical_frame, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
