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
"""
import hashlib
import json
import multiprocessing
import os
from argparse import Namespace
from collections import defaultdict
from typing import Dict, List

import click
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchnlp.download import download_file_maybe_extract
from tqdm import tqdm

from model.tokenizer import Tokenizer
from model.utils import df_emotion_lines_push, df_daily_dialog, load_scenarioSA, augment_dataset

PADDED_INPUTS = ["input_ids"]
MODEL_INPUTS = ["input_ids", "input_lengths", "labels"]
MODEL_INPUTS_NN = ["input_ids", "input_lengths", "labels", "labels_nn"]

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DataModule(pl.LightningDataModule):
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
        sentence: List[int],
        label_encoder: Dict[str, int] = None,
        labels: List[int] = None,
        prepare_labels: bool = True,
        label_nn: int = None,
    ) -> Dict[str, List[int]]:
        if not prepare_labels:
            return {"input_ids": sentence, "input_lengths": len(sentence)}

        label_encoding = [0] * len(label_encoder)
        for l in labels:
            label_encoding[l] = 1

        output = {
            "input_ids": sentence,
            "input_lengths": len(sentence),
            "labels": label_encoding,
        }

        if label_nn != None:
            output["labels_nn"] = label_nn

        return output

    @classmethod
    def build_input_context(
        cls,
        tokenizer: Tokenizer, 
        sentences: List[List[int]],
        label_encoder: Dict[str, int] = None,
        labels: List[int] = None,
        prepare_labels: bool = True, 
        label_nn: int = None,
    ) -> Dict[str, List[int]]:
        if not prepare_labels:
            return {"input_ids": sentence, "input_lengths": len(sentence)}
        
        label_encoding = [0] * len(label_encoder)
        for l in labels:
            label_encoding[l] = 1
        
        input_ids = [tokenizer.bos_index]
        
        for s in sentences:
            input_ids.extend(s)
            input_ids.extend([tokenizer.eos_index])

        # if the input is larger than 512 (bert's max input length), trim.
        if len(input_ids) > 512:
            input_ids = input_ids[:511].extend([tokenizer.eos_index])

        output = {
            "input_ids": input_ids,
            "input_lengths": len(input_ids),
            "labels": label_encoding,
        }

        if label_nn != None:
            output["labels_nn"] = label_nn

        return output


    def _tokenize(self, data: List[Dict[str, str]]):
        if self.hparams.dataset == "goemotions":
            for i in tqdm(range(len(data))):
                data[i]["text"] = self.tokenizer.encode(str(data[i]["text"]))
                data[i]["label"] = [int(label) for label in data[i]["label"].split(",")]

        elif self.hparams.dataset == "emotion-lines" or self.hparams.dataset == "emotion-push":
            for i in tqdm(range(len(data))):
                data[i]["text"] = self.tokenizer.encode(str(data[i]["text"]))
                data[i]["label"] = [data[i]["label"]]
        
        elif self.hparams.dataset == "dailydialog":
            for i in tqdm(range(len(data))):
                data[i]["text"] = self.tokenizer.encode(str(data[i]["text"]))
                data[i]["label"] = [int(label) for label in data[i]["label"]]
        
        elif self.hparams.dataset == "scenariosa":
            for i in tqdm(range(len(data))):
                data[i]["text"] = self.tokenizer.encode(str(data[i]["text"]))
                data[i]["label"] = [data[i]["label"]]

        return data

    def _get_dataset(
        self,
        dataset_path: str,
        data_folder: str = "../data/",
    ):
        """Loads an Emotion Dataset.

        :param dataset_path: Path to a folder containing the training csv, the development csv's
             and the corresponding labels.
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

        # different dataset version in case we are doing augmentation
        if self.hparams.augment_dataset:
            dataset_cache = data_folder + ".dataset_" + str(dataset_hash) + pretrained_model + "_aug"
        else:
            dataset_cache = data_folder + ".dataset_" + str(dataset_hash) + pretrained_model

        if os.path.isfile(dataset_cache):
            click.secho(f"Loading tokenized dataset from cache: {dataset_cache}.")
            return torch.load(dataset_cache)

        dataset_path += "" if dataset_path.endswith("/") else "/"

        # Read Labels
        if self.hparams.dataset == "scenariosa":
            with open(dataset_path + "labels.txt", "r") as fp:
                labels = [line.strip() for line in fp.readlines()]
                label_encoder = {labels[i]: i for i in range(len(labels))}
        else:
            with open(dataset_path + "labels.txt", "r") as fp:
                labels = [line.strip() for line in fp.readlines()]
                label_encoder = {labels[i]: i for i in range(len(labels))}
        
        # Load dataset
        if self.hparams.dataset == "goemotions":
            train = pd.read_csv(dataset_path + "train.tsv", sep="\t")
            valid = pd.read_csv(dataset_path + "valid.tsv", sep="\t")
            test = pd.read_csv(dataset_path + "test.tsv", sep="\t")

        elif self.hparams.dataset == "emotion-lines" or self.hparams.dataset == "emotion-push":

            train = df_emotion_lines_push(dataset_path + "train.json")
            valid = df_emotion_lines_push(dataset_path + "dev.json")
            test = df_emotion_lines_push(dataset_path + "test.json")

            train.replace(label_encoder, inplace=True)
            valid.replace(label_encoder, inplace=True)
            test.replace(label_encoder, inplace=True)

        elif self.hparams.dataset == "dailydialog":
            train = df_daily_dialog(
                dataset_path + "train/" + "dialogues_train.txt", dataset_path + "train/" + "dialogues_emotion_train.txt"
                )
            valid = df_daily_dialog(
                dataset_path + "validation/" + "dialogues_validation.txt", dataset_path + "validation/" + "dialogues_emotion_validation.txt"
                )
            test = df_daily_dialog(
                dataset_path + "test/" + "dialogues_test.txt", dataset_path + "test/" + "dialogues_emotion_test.txt"
                )

        elif self.hparams.dataset == "scenariosa":
            train, valid, test = load_scenarioSA(dataset_path + "InteractiveSentimentDataset/")

        # Augment dataset if needed
        if self.hparams.augment_dataset:
            dataset = {
                "train": augment_dataset(train, label_encoder, self.hparams.pretrained_model).to_dict("records"),
                "valid": valid.to_dict("records"),
                "test": test.to_dict("records"),
            }
        else:
            dataset = {
                "train": train.to_dict("records"),
                "valid": valid.to_dict("records"),
                "test": test.to_dict("records"),
            }

        dataset["label_encoder"] = label_encoder
        # Tokenize
        dataset["train"] = self._tokenize(dataset["train"])
        dataset["valid"] = self._tokenize(dataset["valid"])
        dataset["test"] = self._tokenize(dataset["test"])
        
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
            dataset[name] = [x + [padding] * (max_l - len(x)) for x in dataset[name]]
        return dataset

    def prepare_data(self):
        """
        Lightning DataModule function that will be used to load/download data,
        build inputs with padding and to store everything as TensorDatasets.
        """
        data = self._get_dataset(self.hparams.dataset_path)
        label_encoder = data["label_encoder"]
        del data["label_encoder"]

        click.secho("Building inputs and labels.", fg="yellow")
        datasets = {
            "train": defaultdict(list),
            "valid": defaultdict(list),
            "test": defaultdict(list),
        }

        self.labels_nn = {}

        # load nearest neighbor files
        if self.hparams.retrieval_augmentation:

            if not os.path.isfile(self.hparams.dataset_path + "/train_nn_quorum.json"):
                click.secho(f"{self.hparams.dataset_path}/train_nn.json not found! First run the retrieval command.", fg="red")

            with open(self.hparams.dataset_path + "/train_nn_quorum.json") as train_file:
                self.labels_nn['train'] = json.load(train_file)
            
            with open(self.hparams.dataset_path + "/valid_nn_quorum.json") as valid_file:
                self.labels_nn['valid'] = json.load(valid_file)

            with open(self.hparams.dataset_path + "/test_nn_quorum.json") as test_file:
                self.labels_nn['test'] = json.load(test_file)

        if self.hparams.context:
            for dataset_name, dataset in data.items():
                limit = len(dataset) - 1
                for i, sample in enumerate(dataset):
                    if i >= limit: break

                    # create samples input
                    samples = []
                    samples.append(sample["text"])

                    if i != 0:
                        for turn in range(self.hparams.context_turns):
                            if sample["conv_id"] == dataset[i-(turn+1)]["conv_id"]:
                                samples.append(dataset[i-(turn+1)]["text"]) 

                        if self.hparams.retrieval_augmentation:
                            instance = self.build_input_context(
                                tokenizer=self.tokenizer, 
                                sentences=samples, 
                                label_encoder=label_encoder, 
                                labels=sample["label"], 
                                label_nn=int(self.labels_nn[dataset_name][str(i)])
                                #label_nn=sample["label"][0],
                            )
                        else:    
                            instance = self.build_input_context(
                                tokenizer=self.tokenizer, 
                                sentences=samples, 
                                label_encoder=label_encoder, 
                                labels=sample["label"]
                            )

                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)

        else:
            for dataset_name, dataset in data.items():
                for i, sample in enumerate(dataset):
                    if self.hparams.retrieval_augmentation:
                        instance = self.build_input(
                            tokenizer=self.tokenizer, 
                            sentence=sample["text"], 
                            label_encoder=label_encoder, 
                            labels=sample["label"], 
                            label_nn=int(self.labels_nn[dataset_name][str(i)])
                        )
                    else:    
                        instance = self.build_input(
                            tokenizer=self.tokenizer, 
                            sentence=sample["text"], 
                            label_encoder=label_encoder, 
                            labels=sample["label"]
                        )

                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)

        click.secho("Padding inputs and building tensors.", fg="yellow")
        tensor_datasets = {"train": [], "valid": [], "test": []}

        model_inputs_value = MODEL_INPUTS_NN if self.hparams.retrieval_augmentation else MODEL_INPUTS

        for dataset_name, dataset in datasets.items():
            dataset = self.pad_dataset(dataset, padding=self.tokenizer.pad_index)
            for input_name in model_inputs_value:
                if input_name == "labels":
                    tensor = torch.tensor(dataset[input_name], dtype=torch.float32)
                else:
                    tensor = torch.tensor(dataset[input_name])
                tensor_datasets[dataset_name].append(tensor)

        self.train_dataset = TensorDataset(*tensor_datasets["train"])
        self.valid_dataset = TensorDataset(*tensor_datasets["valid"])
        self.test_dataset = TensorDataset(*tensor_datasets["test"])
        
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
        click.secho(
            "Test dataset (Batch, Candidates, Seq length): {}".format(
                self.test_dataset.tensors[0].shape
            ),
            fg="yellow",
        )

    def prepare_sentiment_accuracy_data(self, data_path, retrieval_augmentation_path):
        dataset_name = data_path.split("/")[2]
        
        if dataset_name == "dailydialog":
            with open("../data/dailydialog/labels.txt", "r") as fp:
                labels = [line.strip() for line in fp.readlines()]
                label_encoder = {labels[i]: i for i in range(len(labels))}
        elif dataset_name == "emotionpush":
            with open("../data/emotionpush/labels.txt", "r") as fp:
                labels = [line.strip() for line in fp.readlines()]
                label_encoder = {labels[i]: i for i in range(len(labels))}

        with open(data_path, "r") as read_file:
            data = json.load(read_file)

        df = pd.DataFrame.from_dict(data)
        df.replace(label_encoder, inplace=True)
        
        dataset = df.to_dict("records")

        if retrieval_augmentation_path != 'False':
            with open(retrieval_augmentation_path) as test_file:
                labels_nn = json.load(test_file)

        # tokenize all sentences
        for i in tqdm(range(len(dataset))):
            
            dataset[i]["reply"] = self.tokenizer.encode(str(dataset[i]["reply"]))
            dataset[i]["gold_reply"] = self.tokenizer.encode(str(dataset[i]["gold_reply"]))
            dataset[i]["gold_label"] = [dataset[i]["gold_label"]]

            # history contains multiple sentences to encode
            for j in (range(len(dataset[i]["history"]))):
                dataset[i]["history"][j] = self.tokenizer.encode(str(dataset[i]["history"][j]))
        
        batch_data = defaultdict(list)

        for i in range(len(dataset)):
            # we're choosing only the last sentence of the history because the model is trained
            # with one sentence as context
            if dataset[i]["history"]:
                sample = [dataset[i]["reply"]] + [dataset[i]["history"][-1]]
            else:
                sample = [dataset[i]["reply"]]

            label_nn = int(labels_nn[str(i)]) if retrieval_augmentation_path != 'False' else None
            
            labels = dataset[i]["gold_label"]

            if isinstance(dataset[i]["gold_label"], list):
                for i, label in enumerate(labels):
                    labels[i] = int(label)

            instance = self.build_input_context(
                tokenizer=self.tokenizer, 
                sentences=sample, 
                label_encoder=label_encoder, 
                labels=labels,
                label_nn=label_nn
            )

            for input_name, input_array in instance.items():
                batch_data[input_name].append(input_array)

        batch_data = self.pad_dataset(batch_data, padding=self.tokenizer.pad_index)

        tensor_dataset = []

        input_name_type = MODEL_INPUTS_NN if retrieval_augmentation_path != 'False' else MODEL_INPUTS

        for input_name in input_name_type:
            if input_name == "labels":
                tensor = torch.tensor(batch_data[input_name], dtype=torch.float32)
            else:
                tensor = torch.tensor(batch_data[input_name])
            tensor_dataset.append(tensor)

        self.sentiment_accuracy_dataset = TensorDataset(*tensor_dataset)
        

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

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
        )

    def sentiment_accuracy_dataloader(self) -> DataLoader:
        """ Function that loads the sentiment accuracy set. """
        return DataLoader(
            self.sentiment_accuracy_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
        )
