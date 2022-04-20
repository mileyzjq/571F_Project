# A Graph Neural Net approach to predict football passing distributions
### 2022 Spring

**Goal:** Based on games from the group stage, predict passing distributions
for games in the round of 16 stage

Current directories
- <code>data/processed/player_data.csv</code> contains retrieved data
- <code>data/processed/trained</code> contains trained model
- <code>data/model/model</code> contains model

Please preinstall snap!

Get features
```bash
$ cd prediction
$ python feature_model.py
```

To run, our python version is 3.7, make sure that the path has been changed accordingly if you do not have python locally. Any project issue, please contact us.
```bash
$ cd prediction
$ sh run_demo.sh
```

The script runs `train.py` so make sure to supply with necessary arguments

```
usage:  train.py [-h] [--input_path INPUT_PATH] [--out_path OUT_PATH]
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

