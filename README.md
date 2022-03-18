# A Graph Neural Net approach to predict football passing distributions
### 2022 Spring

**Goal:** Based on games from the group stage, predict passing distributions
for games in the round of 16 stage

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
$ python pdPrediction_team.py
```
