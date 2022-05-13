import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import pickle


def load_data():
    # Paths
    preprocess_path = 'backend/preprocess/preprocess.csv'

    # Read files
    df = pd.read_csv(preprocess_path)

    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] != df['date'].max()]

    return df.fillna(0)


def model():
    # Load data
    data = load_data()

    # Target and features
    y = data.loc[:, 'total_hits']
    X = data.drop(['date', 'visitor', 'home', 'most_hits', 'total_hits'], axis=1).copy()

    # Split data into testing and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model pipeline
    dt = DecisionTreeRegressor(
        random_state=0,
        splitter='best',
        min_samples_split=500,
        min_samples_leaf=500,
        max_features='sqrt'
    )
    model = Pipeline([('dt', dt)])

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    evaluate_model(X_test, y_test, model)

    pickle.dump(model, open('modeling/total_hits/models/decision_tree.pkl','wb'))


def evaluate_model(X_test, y_test, model):
    # Predict
    predictions = model.predict(X_test)

    # Evaluate
    r2 = round(r2_score(y_test, predictions), 2)
    mse = round(mean_squared_error(y_test, predictions), 2)
    rmse = round(mean_squared_error(y_test, predictions, squared=False), 2)
    mae = round(mean_absolute_error(y_test, predictions), 2)
    evaluations_df = pd.DataFrame({'metric': ['r2', 'mse', 'rmse', 'mae'],
                                   'score': [r2, mse, rmse, mae]})

    print('\nDecision Tree')
    print(evaluations_df)


if __name__ == '__main__':
    tree_model = model()
