# State-of-the-art Conversational AI

This code is based on [HLT-MAIA/lightning-convai](https://github.com/HLT-MAIA/lightning-convai).

## Install:

```bash
virtualenv -p python3.6 env
source convai-env/bin/activate

cd text-generation/lightning-convai-master
pip install -r requirements.txt
```

## Command Line Interface:

### Train:

To set up your training you have to define your model configs. There are several config files already defined in the configs folder.

After defining your hyperparameters run the following command:
```bash
python cli.py train -f configs/example.yaml
```

### Test:

To test your model ability to rank candidate answers and reply to user questions just run the following command:

```bash
python cli.py test --experiment experiments/{experiment_id}/ --test_set data/{test_set}
```


## Interact:
```bash
python cli.py interact --experiment experiments/{experiment_id}/
```

