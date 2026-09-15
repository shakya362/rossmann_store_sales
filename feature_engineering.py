"""Shared feature engineering for the Rossmann sales forecasting project.

Used both at training time (notebooks) and at inference time (Streamlit app)
so the exact same transformations are applied in both places.
"""
import pandas as pd
import numpy as np

NUMERIC_FEATURES = [
    'DayOfWeek', 'Promo', 'SchoolHoliday', 'CompetitionDistance',
    'CompetitionOpenMonths', 'Promo2', 'Promo2OpenWeeks', 'IsPromo2Month',
    'Year', 'Month', 'Day', 'WeekOfYear', 'IsWeekend', 'DaysToHoliday',
    'DaysAfterHoliday', 'StoreTargetEnc'
]
CATEGORICAL_FEATURES = ['StateHoliday', 'StoreType', 'Assortment', 'MonthPeriod']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MONTH_ABBR = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
              7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}


def clean_store_fields(df, max_dist):
    fill_map = {
        'CompetitionDistance': max_dist,
        'Promo2SinceWeek': 0, 'Promo2SinceYear': 0, 'PromoInterval': 'None',
        'CompetitionOpenSinceMonth': 0, 'CompetitionOpenSinceYear': 0,
    }
    return df.fillna(fill_map)


def engineer_features(df, holiday_dates):
    """Add all date-derived, competition-tenure, and promo2 features.
    `df` must already have Store metadata merged in (StoreType, Assortment,
    CompetitionDistance, CompetitionOpenSince[Month/Year], Promo2,
    Promo2Since[Week/Year], PromoInterval) plus Date, DayOfWeek, Promo,
    SchoolHoliday, StateHoliday.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['IsWeekend'] = df['DayOfWeek'].isin([6, 7]).astype(int)
    df['MonthPeriod'] = pd.cut(df['Day'], bins=[0, 10, 20, 31],
                                labels=['Start', 'Mid', 'End']).astype(str)

    hd = pd.DatetimeIndex(sorted(pd.to_datetime(holiday_dates)))
    dates = pd.DatetimeIndex(df['Date'])
    if len(hd) == 0:
        df['DaysToHoliday'] = 365
        df['DaysAfterHoliday'] = 365
    else:
        idx = hd.searchsorted(dates)
        idx_r = np.clip(idx, 0, len(hd) - 1)
        idx_l = np.clip(idx - 1, 0, len(hd) - 1)
        df['DaysToHoliday'] = np.clip((hd[idx_r] - dates).days.to_numpy(), 0, 365)
        df['DaysAfterHoliday'] = np.clip((dates - hd[idx_l]).days.to_numpy(), 0, 365)

    comp_open = pd.to_datetime(
        dict(year=df['CompetitionOpenSinceYear'].replace(0, np.nan),
             month=df['CompetitionOpenSinceMonth'].replace(0, np.nan), day=1),
        errors='coerce'
    )
    comp_months = ((df['Date'].dt.year - comp_open.dt.year) * 12 +
                    (df['Date'].dt.month - comp_open.dt.month))
    df['CompetitionOpenMonths'] = comp_months.clip(lower=0).fillna(0)

    promo2_start = pd.to_datetime(
        df['Promo2SinceYear'].replace(0, np.nan).astype('Int64').astype(str) + '-W' +
        df['Promo2SinceWeek'].replace(0, np.nan).astype('Int64').astype(str).str.zfill(2) + '-1',
        format='%G-W%V-%u', errors='coerce'
    )
    df['Promo2OpenWeeks'] = ((df['Date'] - promo2_start).dt.days / 7).clip(lower=0).fillna(0)

    cur_month_str = df['Month'].map(MONTH_ABBR)
    interval_sets = {iv: set(str(iv).split(',')) for iv in df['PromoInterval'].unique()}
    is_month_in_interval = [m in interval_sets[iv] for m, iv in zip(cur_month_str, df['PromoInterval'])]
    df['IsPromo2Month'] = ((df['Promo2'].values == 1) & np.array(is_month_in_interval)).astype(int)

    return df


def apply_store_target_encoding(df, store_enc_map, global_mean):
    df = df.copy()
    df['StoreTargetEnc'] = df['Store'].map(store_enc_map).fillna(global_mean)
    return df


def to_model_matrix(df):
    """Cast categorical columns and return the exact feature matrix the models expect."""
    df = df.copy()
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype(str).astype('category')
    return df[ALL_FEATURES]
