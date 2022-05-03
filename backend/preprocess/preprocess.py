import pandas as pd
import numpy as np


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


def load_pitching_details(type):
    # Load pitching details
    pitching_details = pd.read_csv('backend/data/pitching/details.csv')
    pitching_details['date'] = pd.to_datetime(pitching_details['date'])
    pitching_details['cwpa'] = pitching_details['cwpa'].apply(lambda x: float(x.strip('%')))
 
    if type =='pitching_starters':
        # Starting pitchers
        df = pitching_details[
            pitching_details['ip'] >= 6
        ].groupby(
            ['date', 'visitor', 'home', 'team']
        ).sum(
        ).reset_index()
        return df

    elif type == 'pitching_bullpen':
        # Bullpen pitchers
        df = pitching_details[
            pitching_details['ip'] < 6
        ].groupby(
            ['date', 'visitor', 'home', 'team']
        ).sum(
        ).reset_index()
        return df

    return None
    


def load_schedules():
    # Load schedules
    schedule = pd.DataFrame()
    for season in range(2022-5, 2022 + 1):
        schedule = schedule.append(pd.read_csv(f'backend/data/schedules/{season}.csv'), ignore_index=True)

    schedule['date'] = pd.to_datetime(schedule['date']) 
    schedule['visitor'] = schedule['visitor'].apply(lambda x: 'Arizona Diamondbacks' if x == "Arizona D'Backs" else x)
    schedule['home'] = schedule['home'].apply(lambda x: 'Arizona Diamondbacks' if x == "Arizona D'Backs" else x)

    return schedule


def feature_engineer(df, type):
    if type == 'batting':
        # Function for feature engineering stats for batting
        df['ba'] = df['h'] / df['ab']
        df['obp'] = (df['h'] + df['bb'] + df['hbp']) / (df['ab'] + df['bb'] + df['hbp'] + df['sf'])
        df['1b'] = df['h'] - (df['2b'] + df['3b'] + df['hr'])
        df['slg'] = (df['1b'] + 2*df['2b'] + 3*df['3b'] + 4*df['hr']) / df['ab']
        df['ops'] = df['obp'] + df['slg']
    elif type == 'pitching_totals' or type == 'pitching_starters' or type == 'pitching_visitor':
        df['era'] = 9 * df['er'] / df['ip']

    return df


def ema(df, type, schedule):

    ema_df = pd.DataFrame()

    for team in df['team'].unique():
        for season in df['date'].dt.year.unique():
            team_df = df[(df['team'] == team) & (df['date'].dt.year == season)].sort_values(by=['date']).copy()
            
            team_schedule = schedule[
                ((schedule['home'] == team) | (schedule['visitor'] == team)) & 
                (schedule['date'].dt.year == season)
            ].drop(['season'], axis=1)

            team_df = pd.merge(
                team_schedule,
                team_df,
                left_on=['date', 'visitor', 'home'],
                right_on=['date', 'visitor', 'home'],
                how='left'
            ).drop_duplicates(
                ['date', 'visitor', 'home', 'team']
            )
            
            team_df['team'] = team_df['team'].shift(1)
            for col in list(set(team_df.columns).difference({'date', 'visitor', 'home', 'team'})):
                team_df[col] = team_df[col].ewm(alpha=.5).mean()
                team_df[col] = team_df[col].shift(1)
            
            ema_df = ema_df.append(team_df.dropna(axis=0), ignore_index=True)
    
    ema_df = pd.merge(
        ema_df, 
        df[['date', 'visitor', 'home', 'team', 'h']],
        left_on=['date', 'visitor', 'home', 'team'], 
        right_on=['date', 'visitor', 'home', 'team'],
        suffixes=('', '_target'),
        how='left'
    ).drop_duplicates(
        ['date', 'visitor', 'home', 'team']
    )

    ema_df = feature_engineer(ema_df, type)

    return ema_df.sort_values(by=['date'])


def preprocess(type):
    if type == 'batting':
        df = load_batting_totals()
    elif type == 'pitching_totals':
        df = load_pitching_totals()
    elif type == 'pitching_starters':
        df = load_pitching_details('pitching_starters')
    elif type == 'pitching_bullpen':
        df = load_pitching_details('pitching_bullpen')
    

    schedule = load_schedules()
    ema_df = ema(df, type, schedule)
        
    return ema_df[['date', 'visitor', 'home', 'team', 'h', 'h_target']]


def load_preprocessed_data():
    batting_df = preprocess('batting')
    pitching_df = preprocess('pitching_totals')

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

    df['most_hits'] = np.where(
        df['h_target_batting_home'] > df['h_target_batting_visitor'], 1, 0
    )

    df['total_hits'] = df['h_target_batting_home'] + df['h_target_batting_visitor']

    df = df.drop(
        [
            'team_home', 'team_visitor', 
            'h_target_batting_home', 'h_target_batting_visitor', 
            'h_target_pitching_home', 'h_target_pitching_visitor'
        ], 
        axis=1
    )

    return df


if __name__ == '__main__':
    df = load_preprocessed_data()
    df.to_csv('backend/preprocess/preprocess.csv', index=False)