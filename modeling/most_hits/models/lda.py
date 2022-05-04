from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

    # Model pipeline
    scaler = StandardScaler()
    lda = LinearDiscriminantAnalysis()
    model = Pipeline([('standardize', scaler), ('lda', lda)])

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'LDA Accuracy: {round(accuracy * 100)}%')
    
    pickle.dump(model, open('modeling/most_hits/models/lda.pkl','wb'))


if __name__ == '__main__':
    model()
