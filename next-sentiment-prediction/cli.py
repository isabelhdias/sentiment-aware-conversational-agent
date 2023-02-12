# -*- coding: utf-8 -*-
r"""
Command Line Interface
=======================
   Commands:
   - train: for Training a new model.
   - interact: Model interactive mode where we can "talk" with a trained model.
   - test: Tests the model ability to rank candidate answers and generate text.
"""
import json
import logging
from model.utils import load_json, df_emotion_lines_push_edit, df_daily_dialog

import click
import multiprocessing
import pytorch_lightning as pl
import torch
import yaml
from collections import defaultdict
from pytorch_lightning import seed_everything
from tqdm import tqdm

from model.data_module import DataModule
from model.emotion_transformer import EmotionTransformer
from model.text_retrieval import TextRetrieval
from trainer import TrainerConfig, build_trainer
from torch.utils.data import DataLoader, TensorDataset


PADDED_INPUTS = ["input_ids"]
MODEL_INPUTS = ["input_ids", "input_lengths", "labels"]
MODEL_INPUTS_NN = ["input_ids", "input_lengths", "labels", "labels_nn"]
MODEL_INPUTS_CLASSIF = ["input_ids", "input_lengths", "labels", "labels_classif"]

@click.group()
def cli():
    pass


@cli.command(name="train")
@click.option(
    "--config",
    "-f",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configure YAML file",
)
def train(config: str) -> None:
    yaml_file = yaml.load(open(config).read(), Loader=yaml.FullLoader)
    
    # Build Trainer
    train_configs = TrainerConfig(yaml_file)
    seed_everything(train_configs.seed)
    trainer = build_trainer(train_configs.namespace())

    # Build Model
    model_config = EmotionTransformer.ModelConfig(yaml_file)
    model = EmotionTransformer(model_config.namespace())
    data = DataModule(model.config, model.tokenizer)
    trainer.num_sanity_val_steps=0
    trainer.fit(model, data)


@cli.command(name="interact")
@click.option(
    "--experiment",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment folder containing the checkpoint we want to interact with.",
)
def interact(experiment: str) -> None:
    """Interactive mode command where we can have a conversation with a trained model
    that impersonates a Vegan that likes cooking and radical activities such as sky-diving.
    """
    model = EmotionTransformer.from_experiment(experiment)
    while 1:
        print("Please write a sentence or quit to exit the interactive shell:")
        # Get input sentence
        input_sentence = input("> ")
        input_sentence = input_sentence.split("~/~")
        if input_sentence == "q" or input_sentence == "quit":
            break
        prediction = model.predict(samples=input_sentence)
        print(json.dumps(prediction[0], indent=3))


@cli.command(name="test")
@click.option(
    "--experiment",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment folder containing the checkpoint we want to interact with.",
)
@click.option(
    "--test_set",
    type=click.STRING,
    required=False,
)
@click.option(
    "--data_path",
    type=click.STRING,
    required=False
)
def test(
    experiment: str,
    test_set: str,
    data_path: str,
) -> None:
    """Testing function where a trained model is tested in its ability to rank candidate
    answers and produce replies.
    """
    model = EmotionTransformer.from_experiment(experiment)
    data = DataModule(model.config, model.tokenizer)
    data.prepare_data()

    # Build a very simple trainer
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        deterministic=True,
        logger=False
    )
    
    if test_set:
        with open(data_path + "labels.txt", "r") as fp:
            labels = [line.strip() for line in fp.readlines()]
            label_encoder = {labels[i]: i for i in range(len(labels))}

        if data_path == "../data/emotionpush/":
            dataset = df_emotion_lines_push_edit(test_set)
            dataset.replace(label_encoder, inplace=True)

        elif test_set == "../data/dailydialog/validation/":
            dataset = df_daily_dialog(test_set + "dialogues_validation.txt", test_set + "dialogues_emotion_validation.txt")
        
        elif test_set == "../data/dailydialog/test/":
            dataset = df_daily_dialog(test_set + "dialogues_test.txt", test_set + "dialogues_emotion_test.txt")

        dataset = dataset.to_dict("records")
        dataset = data._tokenize(dataset)

        dataset_dataloader = defaultdict(list)

        limit = len(dataset) - 2
        #nn = 0

        for i, sample in enumerate(dataset):
            if i >= limit: break

            samples = []
            samples_labels = []

            if i != 0:
                for turn in range(model.hparams.context_turns):
                    if sample["conv_id"] == dataset[i-(turn+1)]["conv_id"]:
                        samples.insert(0, dataset[i-(turn+1)]["text"]) # append to beginning of list 
                        samples_labels.insert(0, dataset[i-(turn+1)]["label"][0])
                
                # when we only have one sentence as context
                if len(samples_labels) < model.hparams.context_turns:
                    while(len(samples_labels) < model.hparams.context_turns):
                        samples_labels.insert(0, len(label_encoder)) 

                if samples:
                    instance = data.build_input_context(
                        tokenizer=model.tokenizer, 
                        sentences=samples, 
                        label_encoder=label_encoder, 
                        labels=sample["label"],
                        #labels_classif=samples_labels,
                        #label_nn=int(model.labels_nn[dataset_name][str(nn)]) if model.hparams.retrieval_augmentation else None,
                    )
                    #nn += 1

                    for input_name, input_array in instance.items():
                        dataset_dataloader[input_name].append(input_array)

        tensor_dataset = []

        if model.hparams.retrieval_augmentation:
            model_inputs_type = MODEL_INPUTS_NN
        elif model.hparams.classification_embeddings:
            model_inputs_type = MODEL_INPUTS_CLASSIF
        else:
            model_inputs_type = MODEL_INPUTS

        dataset = data.pad_dataset(dataset_dataloader, padding=data.tokenizer.pad_index)
        for input_name in model_inputs_type:
            if input_name == "labels":
                tensor = torch.tensor(dataset[input_name], dtype=torch.float32)
            else:
                tensor = torch.tensor(dataset[input_name])
            
            tensor_dataset.append(tensor)
        
        test_dataset = TensorDataset(*tensor_dataset)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=model.hparams.batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
        )

        metrics = trainer.test(model, test_dataloaders=test_dataloader)
        predictions = metrics[0]['y_hat_predictions']
        
        model_name = experiment.split("/")[1]

        f = open(data_path + f"nsp_predictions_{model_name}.txt", "w")

        for pred in predictions:
            f.write(str(pred))

        f.close()

    else:
        trainer.test(model, test_dataloaders=data.val_dataloader())

    


@cli.command(name="retrieval")
@click.option(
    "--data_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the data folder.",
)
@click.option(
    "--method",
    type=click.STRING,
    required=True,
    help="Type of retrieval.",
)
def retrieval(
    data_path: str,
    method: str,
) -> None:
    """Builds faiss index based on the training set.
    """

    retrieval_augmentation = TextRetrieval("simple")

    if method == "faiss":
        click.secho("Building nearest neighbor files using FAISS method", fg="yellow")
        retrieval_augmentation.faiss_method(data_path, method)

    elif method == "brute_force":
        click.secho("Building nearest neighbor files using brute force method", fg="yellow")
        retrieval_augmentation.brute_force_method(data_path, method)

    elif method == "concat":
        click.secho("Building nearest neighbor files using FAISS concat method", fg="yellow")
        retrieval_augmentation.faiss_sentences_concat_method(data_path, method)

    elif method == "quorum":
        click.secho("Building nearest neighbor files using quorum concat method", fg="yellow")
        retrieval_augmentation.quorum_faiss_method(data_path, "concat")

    else:
        print("Unknown method.")

if __name__ == "__main__":
    cli()
