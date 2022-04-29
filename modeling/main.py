
import pickle
import pandas as pd
from datetime import datetime
from most_hits.models.gnb import model as GaussianNB
from most_hits.models.svm import model as SupportVectorMachine
from most_hits.models.log_reg import model as LogisticRegression
from most_hits.models.nn import model as NeuralNetwork
from most_hits.models.rand_forest import model as RandomForest

import sys
sys.path.append("C:\\Users\\tyler\\OneDrive\\Documents\\Python\\MLB\\backend\\preprocess")
from preprocess import load_preprocessed_data


def predict(date):
    # Load models
    gnb = pickle.load(open('modeling/most_hits/models/gnb.pkl','rb'))
    log_reg = pickle.load(open('modeling/most_hits/models/log_reg.pkl','rb'))
    nn = pickle.load(open('modeling/most_hits/models/nn.pkl','rb'))
    random_forest = pickle.load(open('modeling/most_hits/models/rand_forest.pkl','rb'))
    svm = pickle.load(open('modeling/most_hits/models/svm.pkl','rb'))

    # Preprocess data
    df = load_preprocessed_data()


def main():
    # # Dates
    # df = pd.read_csv('backend/data/scores/boxscore.csv')
    # df['date'] = pd.to_datetime(df['date'])
    # dates = df[(datetime(2022, 4, 7) <= df['date']) & (df['date'] <= df['date'].max())]['date'].unique()
    # for date in dates:
    #     print(date)
    RandomForest()
    NeuralNetwork()
    SupportVectorMachine()
    LogisticRegression()
    GaussianNB()



if __name__ == '__main__':
    main()
