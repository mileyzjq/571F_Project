# A Graph Neural Net approach to predict football passing distributions
### 2022 Spring

**Goal:** Based on games from the group stage, predict passing distributions
for games in the round of 16 stage

**To duplicate our results or run experiments of your own with our linear
predictor:** 

Tao predictor

To run
```bash
$ cd prediction
$ python tao_prediction.py
```

and supply with necessary arguments

```
usage: tao_prediction.py [-h] [--input_path INPUT_PATH] [--out_path OUT_PATH]
                         [--weight_path WEIGHT_PATH] [--mode MODE]
                         [--valid_size VALID_SIZE]
                         [--learning_rate LEARNING_RATE] [--epoch EPOCH]
                         [--name NAME]

Soccer

optional arguments:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        The input data
  --out_path OUT_PATH   Path to save the data
  --weight_path WEIGHT_PATH
                        Path to save the data
  --mode MODE           Select whether to train, evaluate, inference the model
  --valid_size VALID_SIZE
                        Proportion of data used as validation set
  --learning_rate LEARNING_RATE
                        Default learning rate
  --epoch EPOCH         epoch number
  --name NAME           Name of the model
```
