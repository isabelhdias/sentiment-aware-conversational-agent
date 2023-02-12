This code was based on [HLT-MAIA/Emotion-Transformer](https://github.com/HLT-MAIA/Emotion-Transformer). 

# Sentiment Classifier

We test the model with the following datasets:
- Emotionlines
- Emotionpush
- DailyDialog

# Install

```bash
virtualenv -p python3.6 env
source env/bin/activate

cd next-sentiment-prediction
pip install -r requirements.txt
```

## Command Line Interface:

### Train:

To set up your training you have to define your model configs. There are several config files already defined in the configs folder.

After defining your hyperparameters run the following command:
```bash
python cli.py train -f configs/example.yaml
```

### Monitor training with Tensorboard:
Launch tensorboard with:

```
tensorboard --logdir="experiments/"
```

## Interact:
```bash
python cli.py interact --experiment experiments/{experiment_id}/
```

## Testing:
```bash
python cli.py test --experiment experiments/{experiment_id}/
```
