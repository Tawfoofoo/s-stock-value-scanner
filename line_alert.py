"""LINE Messaging API notifications for US Stock Value Scanner V4.

Runs after a completed scan in GitHub Actions. Sends a scan report every run,
plus newly eligible value and CDC buy signals. Requires pandas and requests.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

CSV_PATH = Path(os.getenv('SCAN_CSV_PATH', 'data/latest_scan.csv'))
STATE_PATH = Path(os.getenv('LINE_ALERT_STATE_PATH', 'data/line_alert_state.json'))
MIN_MOS = float(os.getenv('LINE_MIN_MOS', '15'))
MAX_AGE_HOURS = float(os.getenv('LINE_MAX_SCAN_AGE_HOURS', '6'))
EXPECTED_STOCKS = int(os.getenv('LINE_EXPECTED_STOCKS', '0'))
API_URL = 'https://api.line.me/v2/bot/message/push'
MAX_TEXT = 4500  # LINE allows 5000 characters per text message.


def is_true(value):
    return str(value).strip().lower() in ('true', '1', 'yes')


def load_state():
    if not STATE_PATH.exists():
        return {'value_alerted': [], 'buy_alerted': []}
    with STATE_PATH.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        'value_alerted': list(data.get('value_alerted', [])),
        'buy_alerted': list(data.get('buy_alerted', [])),
    }


def format_number(value, digits=2):
    number = pd.to_numeric(value, errors='coerce')
    return f'{number:,.{digits}f}' if pd.notna(number) else 'N/A'


def split_lines(lines, limit=MAX_TEXT):
    """Split at line boundaries, preserving all information."""
    chunks = []
    current = ''
    for line in lines:
        if len(line) > limit:
            raise ValueError(f'One LINE report line exceeds {limit} characters')
        candidate = f'{current}\n{line}' if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def send_line(text, token, user_id):
    response = requests.post(
        API_URL,
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
        },
        json={'to': user_id, 'messages': [{'type': 'text', 'text': text}]},
        timeout=25,
    )
    response.raise_for_status()


def stock_line(row):
    return (
        f"{row['Ticker']} | ${format_number(row.get('Price ($)'))} | "
        f"MOS {format_number(row.get('MOS (%)'), 1)}% | "
        f"CDC {row.get('CDC Zone', 'N/A')}"
    )


def main():
    token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', '').strip()
    user_id = os.getenv('LINE_TO_USER_ID', '').strip()
    if not token or not user_id:
        raise RuntimeError('Set LINE_CHANNEL_ACCESS_TOKEN and LINE_TO_USER_ID in GitHub Actions Secrets')
    if not CSV_PATH.exists():
        raise FileNotFoundError(f'Cannot find {CSV_PATH}')

    df = pd.read_csv(CSV_PATH)
    required = {'Ticker', 'Quality Pass', 'Value Pass', 'MOS (%)', 'CDC Buy today', 'Scanned UTC'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')

    scanned = pd.to_datetime(df['Scanned UTC'], errors='coerce', utc=True)
    age = pd.Timestamp.now(tz='UTC') - scanned
    fresh = scanned.notna() & (age >= pd.Timedelta(0)) & (age <= pd.Timedelta(hours=MAX_AGE_HOURS))
    df = df.loc[fresh].copy()
    print(f'Fresh stocks for LINE: {len(df)}', flush=True)
    if df.empty:
        raise RuntimeError('No fresh scan rows; refusing to send a misleading report')

    df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
    df = df[df['Ticker'].str.fullmatch(r'[A-Z0-9.\-]+', na=False)].drop_duplicates('Ticker', keep='last')
    if df.empty:
        raise RuntimeError('No valid tickers in fresh scan')
    if EXPECTED_STOCKS and len(df) < EXPECTED_STOCKS:
        raise RuntimeError(f'Incomplete scan: {len(df)} of {EXPECTED_STOCKS} expected tickers')

    mos = pd.to_numeric(df['MOS (%)'], errors='coerce')
    eligible = df['Quality Pass'].map(is_true) & df['Value Pass'].map(is_true) & (mos >= MIN_MOS)
    buy = df['CDC Buy today'].map(is_true)
    value_df = df.loc[eligible].sort_values('Ticker')
    current_value = set(value_df['Ticker'])
    current_buy = set(df.loc[eligible & buy, 'Ticker'])

    state = load_state()
    previous_value = set(state['value_alerted'])
    previous_buy = set(state['buy_alerted'])
    new_value = sorted(current_value - previous_value)
    new_buy = sorted(current_buy - previous_buy)

    # Preserve state for unobserved tickers in case the scan universe changes.
    observed = set(df['Ticker'])
    next_value = (previous_value - observed) | current_value
    next_buy = (previous_buy - observed) | current_buy

    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [
        '📈 US Stock Value Scanner V4',
        f'📅 Daily Scan Report | {stamp}',
        f'ตรวจสอบ {len(df)} หุ้น | ผ่านเกณฑ์ {len(value_df)} หุ้น',
        f'Quality + Value + MOS ≥ {MIN_MOS:g}%',
        f'หุ้นเข้าเกณฑ์ใหม่ {len(new_value)} | CDC Buy ใหม่ {len(new_buy)}',
        '',
        '💎 หุ้นที่ผ่านเกณฑ์ทั้งหมด',
    ]
    if value_df.empty:
        lines.append('ไม่มีหุ้นผ่านเกณฑ์ในรอบนี้')
    else:
        lines.extend(stock_line(row) for _, row in value_df.iterrows())

    lines.extend(['', '🆕 NEW VALUE ALERT'])
    if new_value:
        for ticker in new_value:
            row = df.loc[df['Ticker'] == ticker].iloc[-1]
            lines.append(stock_line(row))
    else:
        lines.append('ไม่มีหุ้นเข้าเกณฑ์ใหม่')

    lines.extend(['', '🟢 NEW CDC BUY ALERT'])
    if new_buy:
        for ticker in new_buy:
            row = df.loc[df['Ticker'] == ticker].iloc[-1]
            lines.append(stock_line(row))
    else:
        lines.append('ไม่มีสัญญาณ CDC Buy ใหม่')

    lines.extend(['', 'ข้อมูลเพื่อการศึกษา ไม่ใช่คำแนะนำลงทุน'])
    chunks = split_lines(lines)
    print(f'LINE report: {len(value_df)} eligible, {len(new_value)} new value, '
          f'{len(new_buy)} new CDC buy, {len(chunks)} messages', flush=True)
    for index, chunk in enumerate(chunks, 1):
        send_line(chunk, token, user_id)
        print(f'Sent LINE message {index}/{len(chunks)}', flush=True)

    # Update only after every message was accepted by the LINE API.
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = STATE_PATH.with_suffix('.tmp')
    with temp_path.open('w', encoding='utf-8') as f:
        json.dump({
            'value_alerted': sorted(next_value),
            'buy_alerted': sorted(next_buy),
        }, f, ensure_ascii=False, indent=2)
    temp_path.replace(STATE_PATH)
    print('LINE alert state saved', flush=True)


if __name__ == '__main__':
    main()
