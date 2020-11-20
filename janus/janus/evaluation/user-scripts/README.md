# User Scripts

We source pipelines from user scripts found in Kaggle.
We are currently *manually* extracting the pipeline from the script.

Our current criteria for selecting scripts is:
* Must be written in Python
* Must use Scikit-Learn
* Must implement a pipeline (in addition to potentially other code)
* Tabular data: i.e. no image, NLP
  - Known challenges for AutoML
* Classification tasks
* Each row must be an independent observation (i.e. no grouping/analysis required to derive meaningful features)
* Sort by votes
* We use the train.csv (which is what the users had access to for
  developing their model) -- we evaluate on this as well, using CV


Kaggle query:
https://www.kaggle.com/search?q=classify+in%3Acompetitions+tag%3A%22tabular+data%22
* key term: classify, tabular data, competition
  - independent rows rules out the walmart dataset
