import pickle
import pandas as pd


def load_models():
    models = {}
    model_list = ['decision_tree', 'gradient_boosted', 'linear_regression', 'nn', 'random_forest']
    for model in model_list:
        models[model] = pickle.load(open(f'modeling/total_hits/models/{model}.pkl', 'rb'))
    
    return models


def load_data():
    df = pd.read_csv('backend/preprocess/preprocess.csv')
    df['date'] = pd.to_datetime(df['date'])

    df = df[df['date'] == df['date'].max()]

    return df


def predict():
    # Load models
    models = load_models()

    # Load data
    preprocessed = load_data()

    # Make predictions
    new_predictions = preprocessed[['date', 'visitor', 'home', 'total_hits']].copy()
    for model in models:
        new_predictions[model] = models[model].predict(
            preprocessed.drop(
                ['date', 'visitor', 'home', 'most_hits', 'total_hits'], 
                axis=1
            )
        )
        new_predictions[model] = [round(x) for x in new_predictions[model]]
    
    # Save predictions
    predictions = pd.read_csv('backend/data/predictions/total_hits.csv')
    predictions['date'] = pd.to_datetime(predictions['date'])
    predictions = predictions.append(
        new_predictions, 
        ignore_index=True
    ).drop_duplicates(
        ['date', 'visitor', 'home'],
        keep='last'
    )

    predictions.sort_values(
        by=['date']
    ).to_csv(
        'backend/data/predictions/total_hits.csv', 
        index=False
    )


if __name__ == '__main__':
    predict()