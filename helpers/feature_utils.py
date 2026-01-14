import pandas as pd

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    # =========================
    # Binder-related features
    # =========================
    df['SCM Content'] = df['FA'] + df['SF'] + df['GGBFS'] + df['CC']
    df['Binder Content'] = df['C'] + df['SCM Content']
    df['Binder Content'] = df['Binder Content'].replace(0, 1e-6)
    df['W/B Ratio'] = df['W'] / df['Binder Content']
    df['SCM RR'] = df['SCM Content'] / df['Binder Content']
    df['Cement%'] = df['C'] / df['Binder Content']
    df['FA%'] = df['FA'] / df['Binder Content']
    df['SF%'] = df['SF'] / df['Binder Content']
    df['GGBFS%'] = df['GGBFS'] / df['Binder Content']
    df['CC%'] = df['CC'] / df['Binder Content']

    # =========================
    # Aggregate-related features
    # =========================
    df['Total Aggregate Content (kg/m3)'] = df['FAg'] + df['CA'] + df['RCA']
    df['Coarse%'] = (df['CA'] + df['RCA']) / df['Total Aggregate Content (kg/m3)']

    df['RCA RR'] = df['RCA'] / (df['CA'] + df['RCA'] + 1e-6)

    # =========================
    # Additional mix-design ratios
    # =========================
    df['Sp/B Ratio'] = df['SP'] / df['Binder Content']
    df['Sp/W Ratio'] = df['SP'] / df['W']
    df['P/Agg Ratio'] = (df['Binder Content'] + df['W'] + df['SP']) / df['Total Aggregate Content (kg/m3)']
    df['B/Agg Ratio'] = df['Binder Content'] / df['Total Aggregate Content (kg/m3)']
    df['W/S Ratio'] = df['W']/ (df['Binder Content'] + df['Total Aggregate Content (kg/m3)'])
    return df

def step_to_format(step):
    if step == 0: return "%.0f"
    step_str = f"{step:.10f}".rstrip("0")
    if "." in step_str:
        decimals = len(step_str.split(".")[1])
        if decimals == 0: return "%d"
        return f"%.{decimals}f"
    return "%d"
