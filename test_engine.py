"""Offline regression tests for V6 annual FCF handling."""
import unittest
from unittest.mock import patch
import pandas as pd
from engine import annual_fcf_map, fcf_history_or_error, annual_series, dcf

class FCFHistoryTests(unittest.TestCase):
    def setUp(self):
        self.years = ['2025-12-31','2024-12-31','2023-12-31','2022-12-31','2021-12-31']

    def test_four_good_one_old_missing(self):
        m = dict(zip(self.years, [50,40,30,20,None]))
        self.assertEqual(len(fcf_history_or_error(m)), 4)

    def test_three_good_two_old_missing(self):
        m = dict(zip(self.years, [50,40,30,None,None]))
        self.assertEqual(len(fcf_history_or_error(m)), 3)

    def test_internal_gap_rejected(self):
        m = dict(zip(self.years, [50,40,None,20,10]))
        with self.assertRaisesRegex(ValueError, 'found=2'):
            fcf_history_or_error(m)

    def test_latest_missing_rejected(self):
        m = dict(zip(self.years, [None,40,30,20,10]))
        with self.assertRaisesRegex(ValueError, 'found=0'):
            fcf_history_or_error(m)

    def test_negative_fcf_is_observation(self):
        m = dict(zip(self.years, [50,-5,30,None,None]))
        self.assertEqual([v for _,v in fcf_history_or_error(m)], [50,-5,30])

    def test_cfo_capex_fallback_and_zero(self):
        cf = pd.DataFrame({pd.Timestamp('2025-12-31'): [None, 100, -100],
                           pd.Timestamp('2024-12-31'): [0, 100, -30],
                           pd.Timestamp('2023-12-31'): [25, 100, -30]},
                          index=['Free Cash Flow','Operating Cash Flow','Capital Expenditure'])
        m = annual_fcf_map(cf)
        self.assertEqual(m['2025-12-31'], 0)
        self.assertEqual(m['2024-12-31'], 0)
        self.assertEqual(m['2023-12-31'], 25)

    def test_no_fabricated_missing_year(self):
        m = {'2025-12-31': 50, '2023-12-31': 30, '2022-12-31': 20}
        with self.assertRaisesRegex(ValueError, 'found=1'):
            fcf_history_or_error(m)

    def test_dcf_baseline(self):
        self.assertGreater(dcf(1e9, None, None, 1e8, .05, .025, .10), 0)

if __name__ == '__main__':
    unittest.main()
