from statistics import mode
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
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
    scaler = StandardScaler()
    nn = MLPRegressor(
        hidden_layer_sizes=(100, 50, ),
        random_state=0,
        alpha=.0001, 
        batch_size=200, 
        learning_rate='adaptive',
        learning_rate_init=.0001,
        max_iter=200
    )
    model = Pipeline([('scaler', scaler), ('nn', nn)])

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    evaluate_model(X_test, y_test, model)

    pickle.dump(model, open('modeling/total_hits/models/nn.pkl','wb'))


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

    print('\nNN')
    print(evaluations_df)


def hypertune(model, X_train, y_train):
    params = {
        'nn__batch_size': [200, 300],
        'nn__learning_rate': ['adaptive', 'invscaling'],
        'nn__learning_rate_init': [.001, .0001], 
        'nn__alpha': [.1, .01],
        'nn__max_iter': [500]
    }

    nn = GridSearchCV(
        model,
        param_grid=params,
        scoring='neg_mean_absolute_error',
        cv=3, 
        verbose=2,
        n_jobs=10
    )

    nn.fit(X_train, y_train)

    print(nn.best_params_)

    return nn.best_estimator_


if __name__ == '__main__':
    tree_model = model()
