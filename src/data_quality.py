from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


VIETNAM_TIMEZONE = timezone(timedelta(hours=7))
REQUIRED_MARKET_COLUMNS = {'Date', 'Close', 'Target'}


def assess_market_data(data, now=None):
    """Return freshness and integrity checks for the processed market dataset."""
    missing_columns = sorted(REQUIRED_MARKET_COLUMNS.difference(data.columns))
    if data.empty:
        return {
            'status': 'error',
            'issues': ['Không có bản ghi dữ liệu.'],
            'row_count': 0,
            'latest_date': None,
            'calendar_age_days': None,
            'business_days_lag': None,
            'duplicate_dates': 0,
            'invalid_close_count': 0,
            'invalid_target_count': 0,
            'future_date_count': 0,
            'missing_columns': missing_columns,
        }

    if missing_columns:
        return {
            'status': 'error',
            'issues': [f"Thiếu cột bắt buộc: {', '.join(missing_columns)}."],
            'row_count': len(data),
            'latest_date': None,
            'calendar_age_days': None,
            'business_days_lag': None,
            'duplicate_dates': 0,
            'invalid_close_count': 0,
            'invalid_target_count': 0,
            'future_date_count': 0,
            'missing_columns': missing_columns,
        }

    dates = pd.to_datetime(data['Date'], errors='coerce')
    valid_dates = dates.dropna()
    today = (now or datetime.now(VIETNAM_TIMEZONE)).date()
    latest_date = valid_dates.max().date() if not valid_dates.empty else None
    calendar_age_days = max((today - latest_date).days, 0) if latest_date else None
    business_days_lag = (
        len(pd.bdate_range(latest_date + timedelta(days=1), today))
        if latest_date and latest_date < today else 0
    )
    duplicate_dates = int(dates.dropna().duplicated().sum())
    close_values = pd.to_numeric(data['Close'], errors='coerce')
    invalid_close_count = int(
        (close_values.isna() | ~np.isfinite(close_values) | (close_values <= 0)).sum()
    )
    raw_targets = data['Target']
    numeric_targets = pd.to_numeric(raw_targets, errors='coerce')
    invalid_target_count = int(
        (raw_targets.notna() & ~numeric_targets.isin([0, 1])).sum()
    )
    future_date_count = int((valid_dates.dt.date > today).sum())

    issues = []
    if valid_dates.empty:
        issues.append('Không có ngày giao dịch hợp lệ.')
    invalid_date_count = int(dates.isna().sum())
    if invalid_date_count:
        issues.append(f'{invalid_date_count} dòng có ngày không hợp lệ.')
    if duplicate_dates:
        issues.append(f'{duplicate_dates} ngày giao dịch bị trùng.')
    if invalid_close_count:
        issues.append(f'{invalid_close_count} giá đóng cửa bị thiếu hoặc không hợp lệ.')
    if invalid_target_count:
        issues.append(f'{invalid_target_count} nhãn xu hướng không hợp lệ.')
    if future_date_count:
        issues.append(f'{future_date_count} ngày dữ liệu nằm trong tương lai.')
    if business_days_lag is not None and business_days_lag > 5:
        issues.append(f'Dữ liệu trễ {business_days_lag} ngày làm việc.')

    has_integrity_issue = bool(
        invalid_date_count
        or duplicate_dates
        or invalid_close_count
        or invalid_target_count
        or future_date_count
    )
    is_stale = business_days_lag is not None and business_days_lag > 5
    status = 'error' if has_integrity_issue else 'stale' if is_stale else 'ready'
    return {
        'status': status,
        'issues': issues,
        'row_count': len(data),
        'latest_date': latest_date,
        'calendar_age_days': calendar_age_days,
        'business_days_lag': business_days_lag,
        'duplicate_dates': duplicate_dates,
        'invalid_close_count': invalid_close_count,
        'invalid_target_count': invalid_target_count,
        'future_date_count': future_date_count,
        'missing_columns': missing_columns,
    }
