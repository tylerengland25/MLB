from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle

import sys
sys.path.append("C:\\Users\\tyler\\OneDrive\\Documents\\Python\\MLB\\backend\\preprocess")
from preprocess import load_preprocessed_data


def model():
    # Load preprocessed data
    df = load_preprocessed_data()
    X = df.drop(['most_hits', 'total_hits', 'date', 'visitor', 'home'], axis=1)
    y = df['most_hits']

    # Split data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Model pipeline
    scaler = StandardScaler()
    svm = SVC(random_state=1)
    model = Pipeline([('standardize', scaler), ('svm', svm)])

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'SVM Accuracy: {round(accuracy * 100)}%')

    pickle.dump(model, open('modeling/most_hits/models/svm.pkl','wb'))


if __name__ == '__main__':
    model()
