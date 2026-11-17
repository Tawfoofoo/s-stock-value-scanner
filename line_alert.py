"""LINE Messaging API alerts for US Stock Value Scanner V4.

Environment:
  LINE_CHANNEL_ACCESS_TOKEN: Messaging API channel access token
  LINE_TO_USER_ID: LINE user ID (U...) of a user who added the OA as a friend

Run after scan_batch.py, and commit data/line_alert_state.json with the scan results.
Requires: pandas, requests.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

CSV_PATH = Path(os.getenv('SCAN_CSV_PATH', 'data/latest_scan.csv'))
STATE_PATH = Path(os.getenv('LINE_ALERT_STATE_PATH', 'data/line_alert_state.json'))
MIN_MOS = float(os.getenv('LINE_MIN_MOS', '20'))
API_URL = 'https://api.line.me/v2/bot/message/push'


def is_true(value):
    return str(value).strip().lower() in ('true', '1', 'yes')


def load_state():
    if not STATE_PATH.exists():
        return {'value_alerted': [], 'buy_alerted': []}
    with STATE_PATH.open('r', encoding='utf-8') as f:
        state = json.load(f)
    return {
        'value_alerted': list(state.get('value_alerted', [])),
        'buy_alerted': list(state.get('buy_alerted', [])),
    }


def format_number(value, digits=2):
    n = pd.to_numeric(value, errors='coerce')
    return f'{n:,.{digits}f}' if pd.notna(n) else 'N/A'


def send_line(text, token, user_id):
    response = requests.post(
        API_URL,
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
        json={'to': user_id, 'messages': [{'type': 'text', 'text': text}]},
        timeout=25,
    )
    response.raise_for_status()


def main():
    token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', '').strip()
    user_id = os.getenv('LINE_TO_USER_ID', '').strip()
    if not token or not user_id:
        raise RuntimeError('Set LINE_CHANNEL_ACCESS_TOKEN and LINE_TO_USER_ID in GitHub Actions Secrets')
    if not CSV_PATH.exists():
        raise FileNotFoundError(f'Cannot find {CSV_PATH}')

    df = pd.read_csv(CSV_PATH)
    required = {'Ticker', 'Quality Pass', 'Value Pass', 'MOS (%)', 'CDC Buy today'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')
    if df.empty:
        print('CSV is empty; no alerts sent')
        return

    df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
    df = df[df['Ticker'].str.fullmatch(r'[A-Z0-9.\-]+', na=False)].drop_duplicates('Ticker', keep='last')
    mos = pd.to_numeric(df['MOS (%)'], errors='coerce')
    eligible = df['Quality Pass'].map(is_true) & df['Value Pass'].map(is_true) & (mos >= MIN_MOS)
    buy = df['CDC Buy today'].map(is_true)

    state = load_state()
    previously_value = set(state['value_alerted'])
    previously_buy = set(state['buy_alerted'])
    current_value = set(df.loc[eligible, 'Ticker'])
    current_buy = set(df.loc[eligible & buy, 'Ticker'])

    # Only change state for tickers actually present in the CSV. This supports partial batch scans.
    observed = set(df['Ticker'])
    next_value = (previously_value - observed) | current_value
    next_buy = (previously_buy - observed) | current_buy
    new_value = sorted(current_value - previously_value)
    new_buy = sorted(current_buy - previously_buy)

    if not new_value and not new_buy:
        print(f'No new alerts. Rows checked: {len(df)}')
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        lines = [f'📈 US Stock Value Scanner V4', f'ตรวจข้อมูล {len(df)} หุ้น | {stamp}']
        for title, tickers in [('💎 VALUE ALERT', new_value), ('🟢 CDC BUY ALERT', new_buy)]:
            if not tickers:
                continue
            lines.extend(['', title])
            for ticker in tickers:
                row = df.loc[df['Ticker'] == ticker].iloc[-1]
                lines.append(
                    f"{ticker} | ราคา ${format_number(row.get('Price ($)'))} | "
                    f"MOS {format_number(row.get('MOS (%)'), 1)}% | "
                    f"CDC {row.get('CDC Zone', 'N/A')}"
                )
        lines.extend(['', 'ข้อมูลเพื่อการศึกษา ไม่ใช่คำแนะนำลงทุน'])
        message = '\n'.join(lines)
        # LINE text message limit is 5,000 characters. Fail safely rather than silently truncate alerts.
        if len(message) > 4900:
            raise ValueError('Alert exceeds 4,900 characters; split messages before sending')
        send_line(message, token, user_id)
        print(f'Sent LINE alert: {len(new_value)} new value, {len(new_buy)} new CDC buy')

    # Write state only after LINE API has succeeded (or no new alert is needed).
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump({'value_alerted': sorted(next_value), 'buy_alerted': sorted(next_buy)}, f, ensure_ascii=False, indent=2)
    tmp.replace(STATE_PATH)


if __name__ == '__main__':
    main()
