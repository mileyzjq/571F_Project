# A network-assisted approach to predicting passing distributions #
###2022 Spring###

**Data:** UEFA Champions League 2014-15 season. Relevant data can be found in
the data directory in the following subfolders:
* passing distributions: passing_distributions/
* rankings: rankings/
* squad lists: squads/
* scores: scores/
* tactical lineups: lineup/
* individual player statistics: player_statistics/

**Goal:** Based on games from the group stage, predict passing distributions
for games in the round of 16 stage

**Results:** Our linear regression model exceeded our baseline (average past
passing network) by 25.27%

**To duplicate our results or run experiments of your own with our linear
predictor:** 

BaseLine predictor

```bash
$ cd predicted
$ python baseline.py
```

Linear predictor with entire model sharing one set of weights

```bash
$ cd predicted
$ python pdPrediction.py
```

Linear predictor with each team having its own set of weights

```bash
$ cd predicted
$ python pdPrediction-team.py
```
