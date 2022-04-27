import pandas as pd


def load_batting_totals():
    # Load batting totals
    batting_totals = pd.read_csv('backend/data/batting/totals.csv')
    batting_totals['date'] = pd.to_datetime(batting_totals['date'])
    batting_totals['wpa-'] = batting_totals['wpa-'].apply(lambda x: float(x.strip("%")))
    batting_totals['cwpa'] = batting_totals['cwpa'].apply(lambda x: float(x.strip("%")))
    batting_totals = batting_totals.fillna(0)
    batting_totals = batting_totals[batting_totals['ab'] != 0]
    batting_totals = batting_totals[list(set(batting_totals.columns).difference({'obp', 'slg', 'ops', 'wpa', 'ali', 'wpa+', 'wpa-', 'cwpa', 'acli', 're24'}))]

    return batting_totals


def load_pitching_totals():
    # Load pitching totals
    pitching_totals = pd.read_csv('backend/data/pitching/totals.csv')
    pitching_totals['date'] = pd.to_datetime(pitching_totals['date'])
    pitching_totals['cwpa'] = pitching_totals['cwpa'].apply(lambda x: float(x.strip('%')))
    pitching_totals = pitching_totals[list(set(pitching_totals.columns).difference({'era', 'gsc', 'wpa', 'ali', 'cwpa', 'acli', 're24'}))]

    return pitching_totals


def feature_engineer(df, type):
    if type == 'batting':
        # Function for feature engineering stats for batting
        df['ba'] = df['h'] / df['ab']
        df['obp'] = (df['h'] + df['bb'] + df['hbp']) / (df['ab'] + df['bb'] + df['hbp'] + df['sf'])
        df['1b'] = df['h'] - (df['2b'] + df['3b'] + df['hr'])
        df['slg'] = (df['1b'] + 2*df['2b'] + 3*df['3b'] + 4*df['hr']) / df['ab']
        df['ops'] = df['obp'] + df['slg']
    elif type == 'pitching':
        df['era'] = 9 * df['er'] / df['ip']

    return df


def ema(df, type):
    # Function for ema
    temp_df = df.sort_values(by=['team', 'date']).copy()

    sma_df = pd.DataFrame()

    for team in df['team'].unique():
        for season in df['date'].dt.year.unique():
            team_df = temp_df[(temp_df['team'] == team) & (temp_df['date'].dt.year == season)].copy()
            for col in list(set(df.columns).difference({'date', 'visitor', 'home', 'team'})):
                team_df[col] = team_df[col].ewm(alpha=2 / 3).mean()
            
            sma_df = sma_df.append(team_df, ignore_index=True)
    
    sma_df = pd.merge(
        sma_df, 
        df[['date', 'visitor', 'home', 'team', 'h']],
        left_on=['date', 'visitor', 'home', 'team'], 
        right_on=['date', 'visitor', 'home', 'team'],
        suffixes=('', '_target')
    )

    sma_df = feature_engineer(sma_df, type)

    return sma_df.sort_values(by=['date']).dropna(axis=0)


def correlated_columns(df, alpha):
    corr_cols = []
    for col in set(df.columns).difference({'date', 'visitor', 'home', 'team', 'h_target'}):
        if abs(df[col].corr(df['h_target'])) >= alpha:
            corr_cols.append(col)

    return corr_cols


def preprocess(type):
    if type == 'batting':
        df = load_batting_totals()
        ema_df = ema(df, type)

        corr_cols = correlated_columns(ema_df, alpha=.7)
            
        return ema_df[['date', 'visitor', 'home', 'team', 'h_target'] + corr_cols]
    
    elif type == 'pitching':
        df = load_pitching_totals()
        ema_df = ema(df, type)

        corr_cols = correlated_columns(ema_df, alpha=.7)
        
        return ema_df[['date', 'visitor', 'home', 'team', 'h_target'] + corr_cols]


def load_preprocessed_data():
    batting_df = preprocess('batting')
    pitching_df = preprocess('pitching')

    df = pd.merge(
        batting_df,
        pitching_df,
        left_on=['date', 'visitor', 'home', 'team'],
        right_on=['date', 'visitor', 'home', 'team'],
        suffixes=('_batting', '_pitching')
    )

    home_df = df[df['home'] == df['team']].copy()
    visitor_df = df[df['visitor'] == df['team']].copy()

    df = pd.merge(
        home_df, 
        visitor_df,
        left_on=['date', 'visitor', 'home'],
        right_on=['date', 'visitor', 'home'],
        suffixes=('_home', '_visitor')
    )

    return df


if __name__ == '__main__':
    load_preprocessed_data().to_csv('backend/preprocess/preprocess.csv', index=False)