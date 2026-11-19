"""Offline workflow contract regression tests."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent

class WorkflowContractTests(unittest.TestCase):
    def test_report_persisted_before_line(self):
        text = (ROOT / '.github/workflows/daily-scan.yml').read_text()
        self.assertIn('cp data/scan_report.json /tmp/stock-scan-results/', text)
        self.assertIn('cp /tmp/stock-scan-results/scan_report.json data/', text)
        self.assertIn('git add -A -- data/latest_scan.csv data/scan_state.txt data/scan_report.json', text)
        self.assertLess(text.index('Commit scan results'), text.index('Send LINE alert'))

    def test_no_stale_report_after_reset(self):
        text = (ROOT / '.github/workflows/daily-scan.yml').read_text()
        self.assertIn('rm -f data/scan_report.json', text)

if __name__ == '__main__':
    unittest.main()
