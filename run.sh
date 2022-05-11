#!/bin/bash

# python backend/scraping/stats.py

python backend/preprocess/preprocess.py

python modeling/total_hits/models/decision_tree.py
python modeling/total_hits/models/gradient_boosted.py
python modeling/total_hits/models/nn.py
python modeling/total_hits/models/linear_regression.py
python modeling/total_hits/models/random_forest.py

python modeling/total_hits/main.py
