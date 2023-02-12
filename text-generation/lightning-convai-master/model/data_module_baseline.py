# -*- coding: utf-8 -*-
r""" 
DataModule
==========
    The DataModule encapsulates all the steps needed to process data:
    - Download / tokenize
    - Save to disk.
    - Apply transforms (tokenize, pad, batch creation, etc…).
    - Load inside Dataset.
    - Wrap inside a DataLoader.

The most important function to understand inside the DataModule is the `build_input` which 
is responsible by building the inputs that will be used to train the PersonaGPT2 Model.

This function receives a tokenized `persona`, `history` and `reply`, concatenates everything
and builds the language model targets. It also keeps track of the possition of the token used
to represent the entire sequence (the last one).

Example:
>>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
>>> DataModule.build_input(
    tokenizer=tokenizer,
    persona=[[72, 588], [1820, 318]],
    history=[[5303, 1804], [316, 20023]],
    reply=[276, 764],
    lm_labels=False
)
{'input_ids': [50258, 72, 588, 1820, 318, 50260, 5303, 1804, 50261, 316, 20023, 50260, 
276, 764, 50258], 'token_type_ids': [50260, 50260, 50260, 50260, 50260, 50261, 50261, 
50261, 50260, 50260, 50260, 50261, 50261, 50261, 50261], 'mc_token_ids': 14, 'lm_labels': 
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]}

>>> DataModule.build_input(
    tokenizer=tokenizer,
    persona=[[72, 588], [1820, 318]],
    history=[[5303, 1804], [316, 20023]],
    reply=[276, 764],
    lm_labels=True
)
{'input_ids': [50258, 72, 588, 1820, 318, 50260, 5303, 1804, 50261, 316, 20023, 50260, 
276, 764, 50258],  'token_type_ids': [50260, 50260, 50260, 50260, 50260, 50261, 50261, 
50261, 50260, 50260, 50260, 50261, 50261, 50261, 50261], 'mc_token_ids': 14, 'lm_labels': 
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 276, 764, 50258]}
"""
import hashlib
import json
import multiprocessing
import os
from argparse import Namespace
from collections import defaultdict
from itertools import chain
import random
from tqdm import tqdm
from typing import Dict, List

import click
import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from model.tokenizer import Tokenizer
from model.utils import df_emotion_lines_push, df_daily_dialog, load_scenarioSA, load_json
from torchnlp.download import download_file_maybe_extract

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DataModuleBaseline(pl.LightningDataModule):
    """PyTorch Lightning DataModule.

    :param hparams: Namespace with data specific arguments.
    :param tokenizer: Model Tokenizer.

    """

    def __init__(self, hparams: Namespace, tokenizer: Tokenizer):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = tokenizer

    @classmethod
    def build_input(
        cls,
        tokenizer: Tokenizer,
        history: List[List[int]],
        reply: List[int] = [],
        sentiment_representation: str = None,
        lm_labels: bool = False,
    ) -> Dict[str, List[int]]:
        """Builds a model input.

        :param persona: List of persona sentences tokenized.
        :param history: List of history sentences tokenizes.
        :param reply: Tokenized answer.
        :param lm_labels: Flag to build LM labels for ground-truth replies.

        :return: Dictionary with model inputs.
        """
        bos, eos, speaker1, speaker2 = (
            tokenizer.bos_index,
            tokenizer.eos_index,
            tokenizer.speaker1_index,
            tokenizer.speaker2_index,
        )

        if sentiment_representation:
            # if we're using multiple sentences as sentiment representation
            if isinstance(sentiment_representation[0], list):
                sequence = (
                    [[bos] + list(chain(*sentiment_representation))]  # concats sentiment representation
                    + history  # concats history
                    + [reply + [eos]]  # concats reply
                )
            else:
                sequence = (
                    [[bos] + sentiment_representation]  # concats sentiment representation
                    + history  # concats history
                    + [reply + [eos]]  # concats reply
                )
            
        else:
            sequence = (
                [[bos]] + history  # concats history
                + [reply + [eos]]  # concats reply
            )
        
        sequence = [sequence[0]] + [
            [speaker2 if (len(sequence) - i) % 2 else speaker1] + s
            for i, s in enumerate(sequence[1:])
        ]

        instance = {
            "input_ids": list(chain(*sequence)),
            "token_type_ids": [
                speaker2 if i % 2 else speaker1
                for i, s in enumerate(sequence)
                for _ in s
            ],
        }

        # to calculate loss
        instance["lm_labels"] = [-100] * len(instance["input_ids"])
        if lm_labels:
            instance["lm_labels"] = (
                ([-100] * sum(len(s) for s in sequence[:-1]))
                + [-100]
                + sequence[-1][1:]
            )
        return instance

    def _tokenize(self, data):
        
        if self.hparams.dataset == "emotionlines" or self.hparams.dataset == "emotionpush":
            for i in tqdm(range(len(data))):
                data[i]["text"] = self.tokenizer.encode(str(data[i]["text"]))
                data[i]["gold_label"] = data[i]["label"]
                data[i]["label"] = self.tokenizer.encode(str(data[i]["label"]))
        
        elif self.hparams.dataset == "dailydialog":
            for i in tqdm(range(len(data))):
                data[i]["text"] = self.tokenizer.encode(str(data[i]["text"]))
                data[i]["gold_label"] = data[i]["label"]
                data[i]["label"] = self.tokenizer.encode(str(data[i]["label"]))
        
        elif self.hparams.dataset == "scenariosa":
            for i in tqdm(range(len(data))):
                data[i]["text"] = self.tokenizer.encode(str(data[i]["text"]))
                data[i]["gold_label"] = data[i]["label"]
                data[i]["label"] = self.tokenizer.encode(str(data[i]["label"]))

        return data

    def _get_dataset(
        self,
        dataset_path: str = "",
        data_folder: str = "data/",
    ):
        """TODO: Downloads PersonaChat corpus from S3 if no dataset_path is provided.

        :param dataset_path: Path to a json file containing the train and validation dataset.
        :param data_folder: Folder used to store data.

        :return: Returns a dictionary with the training and validation data.
        """
        if not os.path.isdir(dataset_path):
            click.secho(f"{dataset_path} not found!", fg="red")

        dataset_hash = (
            int(hashlib.sha256(dataset_path.encode("utf-8")).hexdigest(), 16) % 10 ** 8
        )

        # To avoid using cache for different models
        # split(/) for google/electra-base-discriminator
        pretrained_model = (
            self.hparams.pretrained_model.split("/")[1]
            if "/" in self.hparams.pretrained_model
            else self.hparams.pretrained_model
        )

        dataset_cache = data_folder + ".dataset_" + str(dataset_hash) + pretrained_model + "baseline"

        if os.path.isfile(dataset_cache):
            click.secho(f"Loading tokenized dataset from cache: {dataset_cache}.")
            return torch.load(dataset_cache)

        dataset_path += "" if dataset_path.endswith("/") else "/"

        with open(dataset_path + "labels.txt", "r") as fp:
            labels = [line.strip() for line in fp.readlines()]
            label_encoder = {labels[i]: i for i in range(len(labels))}

        if self.hparams.dataset == "emotionlines" or self.hparams.dataset == "emotionpush":

            train = df_emotion_lines_push(dataset_path + "train.json")
            valid = df_emotion_lines_push(dataset_path + "dev.json")

        elif self.hparams.dataset == "dailydialog":
            train = df_daily_dialog(
                dataset_path + "train/" + "dialogues_train.txt", dataset_path + "train/" + "dialogues_emotion_train.txt"
                )
            valid = df_daily_dialog(
                dataset_path + "validation/" + "dialogues_validation.txt", dataset_path + "validation/" + "dialogues_emotion_validation.txt"
                )

        elif self.hparams.dataset == "scenariosa":
            train, valid = load_scenarioSA(dataset_path + "InteractiveSentimentDataset/")

        dataset = {
            "train": train.to_dict("records"),
            "valid": valid.to_dict("records"),
        }

        dataset["label_encoder"] = label_encoder
        
        # Tokenize
        click.secho("Running tokenization: This might take some time!", fg="yellow")
        dataset["train"] = self._tokenize(dataset["train"])
        dataset["valid"] = self._tokenize(dataset["valid"])

        torch.save(dataset, dataset_cache)

        return dataset

    @classmethod
    def pad_dataset(
        cls, dataset: dict, padding: int = 0, padded_inputs: List[str] = PADDED_INPUTS
    ):
        """
        Pad the dataset.
        NOTE: This could be optimized by defining a Dataset class and
        padding at the batch level, but this is simpler.

        :param dataset: Dictionary with sequences to pad.
        :param padding: padding index.
        :param padded_inputs:
        """
        max_l = max(len(x) for x in dataset["input_ids"])
        for name in padded_inputs:
            dataset[name] = [
                x + [padding if name != "lm_labels" else -100] * (max_l - len(x))
                for x in dataset[name]
            ]
        return dataset

    def prepare_data(self):
        """
        Lightning DataModule function that will be used to load/download data,
        build inputs with padding and to store everything as TensorDatasets.
        """
        emotion_dataset = self._get_dataset(self.hparams.dataset_path)
        label_encoder = emotion_dataset["label_encoder"]
        del emotion_dataset["label_encoder"]

        # Read words set file and tokenize it
        if self.hparams.sentiment_representation == "words-set":
            words_set = load_json(self.hparams.words_set_path)
            for _, sent_set in words_set.items():
                for i, word in enumerate(sent_set):
                    sent_set[i] = self.tokenizer.encode(word)

        if self.hparams.sentiment_representation == "random-sample":
            sentences_per_sentiment_set = df_emotion_lines_push(self.hparams.dataset_path + "train.json")

            for i in range(len(sentences_per_sentiment_set)):
                sentences_per_sentiment_set.loc[i, "text"] = self.tokenizer.encode(sentences_per_sentiment_set.loc[i, "text"])

        if self.hparams.sentiment_representation == "sentiment-sentences":
            if self.hparams.dataset == "emotionpush":
                sentiment_sentences = load_json("data/sentiment_sentences.json")
            elif self.hparams.dataset == "dailydialog":
                sentiment_sentences = load_json("data/sentiment_sentences_dailydialog.json")

            for sentiment, sentences in sentiment_sentences.items():
                for i, sentence in enumerate(sentences):
                    sentiment_sentences[sentiment][i] = self.tokenizer.encode(sentence)


        click.secho("Building inputs and labels.", fg="yellow")
        datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
        for dataset_name, dataset in emotion_dataset.items():
            # make sure we don't access positions that don't exist
            limit = len(dataset) - self.hparams.max_history
            for i, sample in enumerate(dataset):
                if i >= limit: break

                history = []
                reply = sample["text"]

                if i > self.hparams.max_history - 1:
                    for turn in range(self.hparams.max_history):
                        if sample["conv_id"] == dataset[i-(turn+1)]["conv_id"]:
                            history.append(dataset[i-(turn+1)]["text"])

                    if self.hparams.sentiment_representation == "tag": 
                        # this label is already encoded
                        sentiment_representation = sample["label"]
                    
                    elif self.hparams.sentiment_representation == "words-set": 
                        if words_set[sample["gold_label"]]:
                            sentiment_representation = random.choices(words_set[sample["gold_label"]], k=1)
            
                            # flatten list
                            sentiment_representation = [item for sublist in sentiment_representation for item in sublist]
                        else: 
                            sentiment_representation = None
                    
                    elif self.hparams.sentiment_representation == "random-sample":
                        # selects the slice of the dataframe correspondent to the gold label
                        sentiment_df = sentences_per_sentiment_set.loc[sentences_per_sentiment_set['label'] == sample["gold_label"]]
                        # selects a random sentence from the slice
                        sentiment_representation = sentiment_df.sample()['text'].values[0]

                    elif self.hparams.sentiment_representation == "sentiment-sentences":
                        try:
                            sentiment_representation = sentiment_sentences[sample["gold_label"]]
                        except:
                            import pdb; pdb.set_trace()
                    
                    else:
                        sentiment_representation = None
                    
                    instance = self.build_input(
                        tokenizer=self.tokenizer, 
                        history=history, 
                        reply=reply,
                        sentiment_representation=sentiment_representation,
                        lm_labels=True
                    )

                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)
        
        click.secho("Padding inputs and building tensors.", fg="yellow")
        tensor_datasets = {"train": [], "valid": []}
        for dataset_name, dataset in datasets.items():
            dataset = self.pad_dataset(dataset, padding=self.tokenizer.pad_index)

            for input_name in MODEL_INPUTS:
                tensor = torch.tensor(dataset[input_name])
                tensor_datasets[dataset_name].append(tensor)

        self.train_dataset = TensorDataset(*tensor_datasets["train"])
        self.valid_dataset = TensorDataset(*tensor_datasets["valid"])
        click.secho(
            "Train dataset (Batch, Candidates, Seq length): {}".format(
                self.train_dataset.tensors[0].shape
            ),
            fg="yellow",
        )
        click.secho(
            "Valid dataset (Batch, Candidates, Seq length): {}".format(
                self.valid_dataset.tensors[0].shape
            ),
            fg="yellow",
        )

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
        )