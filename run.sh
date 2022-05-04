#!/bin/bash

python backend/scraping/stats.py

python backend/preprocess/preprocess.py

python modeling/most_hits/models/log_reg.py
python modeling/most_hits/models/nn.py
python modeling/most_hits/models/svm.py
python modeling/most_hits/models/rand_forest.py
python modeling/most_hits/models/gnb.py
python modeling/most_hits/models/lda.py

python modeling/most_hits/models/main.py

python modeling/most_hits/models/evaluate.py
