from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle
import pandas as pd


def model():
    # Load preprocessed data
    df = pd.read_csv('backend/preprocess/preprocess.csv').dropna(axis=0)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year < 2022]
    X = df.drop(['most_hits', 'total_hits', 'date', 'visitor', 'home'], axis=1)
    y = df['most_hits']

    # Split data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Load models
    models = {}
    model_list = ['gnb', 'log_reg', 'nn', 'svm']
    for model in model_list:
        models[model] = pickle.load(open(f'modeling/most_hits/models/{model}.pkl', 'rb'))

    models = [(key, models[key]) for key in models]

    # Model pipeline
    scaler = StandardScaler()
    clf = VotingClassifier(
        estimators=models,
        voting='hard'
        )
    model = Pipeline([('standardize', scaler), ('vote', clf)])

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Voting Accuracy: {round(accuracy * 100)}%')

    pickle.dump(model, open('modeling/most_hits/models/voting.pkl','wb'))


if __name__ == '__main__':
    model()
