import pickle
import pandas as pd


def load_models():
    models = {}
    model_list = ['gnb', 'log_reg', 'nn', 'rand_forest', 'svm']
    for model in model_list:
        models[model] = pickle.load(open(f'modeling/most_hits/models/{model}.pkl', 'rb'))
    
    return models


def load_data():
    df = pd.read_csv('backend/preprocess/preprocess.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'].dt.year >= 2022) & (df['date'] != df['date'].max())]

    return df


def predict():
    # Load models
    models = load_models()

    # Load data
    preprocessed = load_data()

    # Make predictions
    predictions = preprocessed[['date', 'visitor', 'home', 'most_hits']].copy()
    for model in models:
        predictions[model] = models[model].predict(
            preprocessed.drop(
                ['date', 'visitor', 'home', 'most_hits', 'total_hits'], 
                axis=1
            )
        )
    
    # Save predictions
    predictions.to_csv('backend/data/predictions/most_hits.csv', index=False)


if __name__ == '__main__':
    predict()
