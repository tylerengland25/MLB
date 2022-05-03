import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def calc_profit(odds):
    if int(odds) > 0:
        return int(odds) / 100
    else:
        return 100 / abs(int(odds))


def load_data():
    # Load predictions
    predictions = pd.read_csv('backend/data/predictions/most_hits.csv')
    predictions['date'] = pd.to_datetime(predictions['date'])

    # Merge odds
    odds = pd.read_excel('backend/data/odds/most_hits.xlsx').dropna(axis=0)
    odds['date'] = pd.to_datetime(odds['date'])
    df = pd.merge(
        odds,
        predictions,
        left_on=['date', 'visitor', 'home'],
        right_on=['date', 'visitor', 'home']
    )

    # Calculate profit
    for model in ['gnb', 'log_reg', 'nn', 'rand_forest', 'svm']:
        df[f'{model}_potential_profit'] = np.where(
            df[model], 
            df['home_odds'].apply(calc_profit), 
            df['visitor_odds'].apply(calc_profit)
        )

        df[f'{model}_profit'] = np.where(
            df[model] == df['most_hits'],
            df[f'{model}_potential_profit'],
            -1
        )

        df[f'{model}_hit_bust'] = df['most_hits'] == df[model]

    return df


def evaluate(df):
    # Graph and print totals
    sns.set(rc={"figure.figsize":(10, 10)})
    sns.set_palette('bright', 8)
    plt.xlabel('date', fontsize=20)
    plt.ylabel('units', fontsize=20)
    plt.xticks(fontsize=15, rotation=45)
    plt.yticks(fontsize=15)

    totals = {'model': [], 'record': [], 'accuracy': [], 'profit': []}
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'black']
    models = ['gnb', 'log_reg', 'nn', 'rand_forest', 'svm']
    for model, col in zip(models, colors):
        # Line graph
        sns.lineplot(
            data=df.groupby(['date']).sum(), 
            x='date', 
            y=df.groupby(['date']).sum()[f'{model}_profit'].cumsum(), 
            color=col, 
            label=model
        )
        
        # Totals
        profit = round(df[f'{model}_profit'].sum(), 2)
        correct = df[f'{model}_hit_bust'].sum()
        wrong = df[f'{model}_hit_bust'].count() - df[f'{model}_hit_bust'].sum()
        
        totals['model'].append(model)
        totals['record'].append(f'{correct} - {wrong}')
        totals['accuracy'].append(f'{round((correct / (correct + wrong) * 100))}%')
        totals['profit'].append(round(profit, 2))
        

    totals = pd.DataFrame(totals).sort_values(['profit', 'accuracy'], ascending=False)
    print(totals)


def main():
    df = load_data()
    evaluate(df)


if __name__ == '__main__':
    main()
