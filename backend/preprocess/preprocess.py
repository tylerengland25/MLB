from turtle import home
import pandas as pd
import numpy as np


def load_schedule():
    # Load schedules
    schedule = pd.DataFrame()
    for season in range(2022-5, 2023):
        schedule = schedule.append(pd.read_csv(f'backend/data/schedules/{season}.csv'), ignore_index=True)

    schedule['date'] = pd.to_datetime(schedule['date'])
    schedule = schedule.drop_duplicates(['date', 'visitor', 'home'])
    
    return schedule


def load_scores():
    # Load scores
    scores = pd.read_csv('backend/data/scores/boxscore.csv')
    scores['date'] = pd.to_datetime(scores['date'])
    scores = scores.drop_duplicates(['date', 'visitor', 'home', 'team'])

    scores = scores[['date', 'visitor', 'home', 'team', 'H']]
    scores['opponent'] = np.where(scores['team'] == scores['home'], scores['visitor'], scores['home'])
    scores['season'] = scores['date'].dt.year

    # Load schedule
    schedule = load_schedule()
    next_slate = schedule[schedule['date'] > scores['date'].max()]['date'].min()
    schedule = schedule[schedule['date'] == next_slate]

    home_sched = schedule.copy()
    home_sched['team'] = home_sched['home']
    home_sched['opponent'] = home_sched['visitor']

    visitor_sched = schedule.copy()
    visitor_sched['team'] = visitor_sched['visitor']
    visitor_sched['opponent'] = visitor_sched['home']

    schedule = home_sched.append(visitor_sched, ignore_index=True)

    df = pd.merge(
        scores, 
        scores, 
        left_on=['date', 'visitor', 'home', 'team'],
        right_on=['date', 'visitor', 'home', 'opponent'],
        suffixes=('_scored', '_allowed')
    ).drop(
        ['team_allowed', 'opponent_allowed', 'season_allowed'],
        axis=1
    ).rename(
        {'team_scored': 'team', 'opponent_scored': 'opponent', 'season_scored': 'season'},
        axis=1
    ).dropna(
        subset=['team'],
        axis=0
    )

    df = df.append(schedule, ignore_index=True)

    df = df.set_index(
        ['team']
    ).sort_index()

    return df


def sma(num_games, df):
    ma_df = pd.DataFrame()
    for index in df.index.unique():
        matchup_df = df.loc[index, :].sort_values(by=['date']).copy()

        matchup_df['H_scored_avg'] = matchup_df['H_scored'].rolling(num_games, closed='left', min_periods=num_games).mean()
        matchup_df['H_scored_std'] = matchup_df['H_scored'].rolling(num_games, closed='left', min_periods=num_games).std()

        matchup_df['H_allowed_avg'] = matchup_df['H_allowed'].rolling(num_games, closed='left', min_periods=num_games).mean()
        matchup_df['H_allowed_std'] = matchup_df['H_allowed'].rolling(num_games, closed='left', min_periods=num_games).std()

        ma_df = ma_df.append(matchup_df.reset_index(), ignore_index=True)

    ma_df = ma_df.dropna(subset=['H_scored_avg'], axis=0)

    cols = ['date', 'visitor', 'home', 'H_scored_avg', 'H_scored_std', 'H_allowed_avg', 'H_allowed_std']
    home_df = ma_df[ma_df['home'] == ma_df['team']][cols]
    visitor_df = ma_df[ma_df['visitor'] == ma_df['team']][cols]

    df_ma = pd.merge(
        home_df, 
        visitor_df,
        left_on=['date', 'visitor', 'home'],
        right_on=['date', 'visitor', 'home'],
        suffixes=('_home', '_visitor')
    )
    
    return df_ma


def load_outcome(df):
    df = df.reset_index()
    home_df = df[df['home'] == df['team']][['date', 'visitor', 'home', 'H_scored', 'H_allowed']]
    visitor_df = df[df['visitor'] == df['team']][['date', 'visitor', 'home', 'H_scored', 'H_allowed']]

    df_actual = pd.merge(
        home_df, 
        visitor_df,
        left_on=['date', 'visitor', 'home'],
        right_on=['date', 'visitor', 'home'],
        suffixes=('_home', '_visitor')
    )

    df_actual['most_hits'] = np.where(df_actual['H_scored_home'] > df_actual['H_allowed_home'], 1, 0)
    df_actual['total_hits'] = df_actual['H_scored_home'] + df_actual['H_allowed_home']

    return df_actual


def merge_ma_outcome(ma, outcome):
    df = pd.merge(
        outcome,
        ma, 
        left_on=['date', 'visitor', 'home'],
        right_on=['date', 'visitor', 'home'],
        suffixes=('_actual', '_expected')
    )

    df = df[
        [
            'date', 'visitor', 'home', 'most_hits', 'total_hits', 
            'H_scored_avg_home', 'H_scored_std_home', 
            'H_allowed_avg_home', 'H_allowed_std_home',
            'H_scored_avg_visitor', 'H_scored_std_visitor', 
            'H_allowed_avg_visitor', 'H_allowed_std_visitor'
        ]
    ]

    return df.sort_values(by=['date'])


def load_preprocessed_data():
    df = load_scores()

    df_ma = sma(20, df)
    df_outcome = load_outcome(df)

    df = merge_ma_outcome(df_ma, df_outcome)

    return df


if __name__ == '__main__':
    df = load_preprocessed_data()
    df.to_csv('backend/preprocess/preprocess.csv', index=False)