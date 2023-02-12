# -*- coding: utf-8 -*-
r""" 
EmotionTransformer Model
==================
    Hugging-face Transformer Model implementing the PyTorch Lightning interface that
    can be used to train an Emotion Classifier.
"""
import multiprocessing
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple

import click
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning.metrics.functional import accuracy
from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torchnlp.utils import collate_tensors, lengths_to_mask
from transformers import AdamW, AutoModel
from tqdm import tqdm

from model.data_module import DataModule
from model.tokenizer import Tokenizer
from model.utils import average_pooling, mask_fill, max_pooling, error_analysis, micro_f1
from utils import Config

POLARITY = ["negative", "neutral", "positive"]

EKMAN = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

EMOTIONLINESPUSH = ["anger", "disgust", "fear", "joy", "neutral", "non-neutral", "sadness", "surprise"]

DAILYDIALOG = ["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

GOEMOTIONS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


class EmotionTransformer(pl.LightningModule):
    """Hugging-face Transformer Model implementing the PyTorch Lightning interface that
    can be used to train an Emotion Classifier.

    :param hparams: ArgumentParser containing the hyperparameters.
    """

    class ModelConfig(Config):
        """The ModelConfig class is used to define Model settings.

        ------------------ Architecture --------------------- 
        :param pretrained_model: Pretrained Transformer model to be used.
        :param pooling: Pooling method for extracting sentence embeddings 
            (options: cls, avg, max, cls+avg)
        
        ----------------- Tranfer Learning --------------------- 
        :param nr_frozen_epochs: number of epochs where the `encoder` model is frozen.
        :param encoder_learning_rate: Learning rate to be used to fine-tune parameters from the `encoder`.
        :param learning_rate: Learning Rate used during training.
        :param layerwise_decay: Learning rate decay for to be applied to the encoder layers.

        ----------------------- Data --------------------- 
        :param dataset_path: Path to a json file containing our data.
        :param labels: Label set (options: `ekman`, `goemotions`)
        :param batch_size: Batch Size used during training.
        """

        pretrained_model: str = "roberta-base"
        pooling: str = "avg"
        nr_layers_pooling: int = 1
        dropout_prob: float = 0.1

        # Optimizer
        nr_frozen_epochs: int = 1
        encoder_learning_rate: float = 1.0e-5
        learning_rate: float = 5.0e-5
        layerwise_decay: float = 0.95

        # Data configs
        dataset_path: str = ""
        labels: str = "ekman"
        dataset: str = "goemotions"
        context: bool = False
        context_turns: int = 1
        augment_dataset: bool = False
        undersample: bool = False

        # Training details
        batch_size: int = 8

        # Classification setup
        classif_setup: str = "linear"

        # Retrieval augmentation
        retrieval_augmentation: bool = False
        init_sentiment_embeddings: bool = False
        sentiment_embeddings_size: int = 128
        sentiment_lambda: bool = False
        sentiment_representation: str = "simple"

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.config = hparams
        self.save_hyperparameters(self.config)

        if self.config.augment_dataset and self.config.context:
            # TODO: think about how to implement context + augmentation
            raise NotImplementedError

        self.transformer = AutoModel.from_pretrained(self.config.pretrained_model)
        self.tokenizer = Tokenizer(self.config.pretrained_model, self.config.context)
        
        # Resize embeddings to include the added tokens
        self.transformer.resize_token_embeddings(self.tokenizer.vocab_size)

        self.encoder_features = self.transformer.config.hidden_size
        self.num_layers = self.transformer.config.num_hidden_layers + 1

        self.dropout = torch.nn.Dropout(self.config.dropout_prob)

        if self.config.labels == "ekman":
            self.label_encoder = {EKMAN[i]: i for i in range(len(EKMAN))}
        elif self.config.labels == "goemotions":
            self.label_encoder = {GOEMOTIONS[i]: i for i in range(len(GOEMOTIONS))}
        elif self.config.labels == "ekman_nn":
            self.label_encoder = {EMOTIONLINESPUSH[i]: i for i in range(len(EMOTIONLINESPUSH))}
        elif self.config.labels == "ekman_dd":
            self.label_encoder = {DAILYDIALOG[i]: i for i in range(len(DAILYDIALOG))}
        elif self.config.labels == "polarity":
            self.label_encoder = {POLARITY[i]: i for i in range(len(POLARITY))}
        else:
            raise Exception("unrecognized label set: {}".format(self.config.labels))
        
        # If pooling is biLSTM
        if self.config.pooling == "bilstm":
            self.biLSTM = nn.LSTM(
                input_size=self.encoder_features * self.config.nr_layers_pooling, 
                hidden_size=(self.encoder_features * self.config.nr_layers_pooling) // 2, 
                num_layers=2,
                dropout=0.2,
                batch_first=True,
                bidirectional=True,
            )
        
        # only for the validation sanity check
        if self.config.retrieval_augmentation:
            if self.config.init_sentiment_embeddings:
                self.sentiment_embeddings = nn.Embedding(len(self.label_encoder), self.transformer.config.hidden_size)
                if self.config.sentiment_embeddings_size != 1024:
                    self.redimension_embeddings = nn.Linear(
                        in_features=self.encoder_features, 
                        out_features=self.config.sentiment_embeddings_size
                    )

            else:
                # if we don't initialize the sentiment embedding matrix, create a smaller embedding matrix
                self.sentiment_embeddings = nn.Embedding(len(self.label_encoder), self.config.sentiment_embeddings_size)

            if self.config.sentiment_lambda:
                self.sentiment_lambda = nn.Embedding(len(self.label_encoder), 1)
                # init weights to 1
                self.sentiment_lambda.weight.data.fill_(1)
        
        # Classification head
        self.classification_head = self.set_classification_head()

        self.loss = nn.BCEWithLogitsLoss()

        if self.config.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = self.config.nr_frozen_epochs


    def init_sentiment_representations(self, train_dataloader):
        device = self.transformer.device
        # put model in evaluation mode
        self.transformer.eval()

        embeddings = []
        gold_labels = []

        for i, batch in tqdm(enumerate(train_dataloader), desc="computing embeddings", total=len(train_dataloader)):
            input_ids, input_lengths, labels, labels_nn = batch

            # no grad so it doesn't save gradients (takes up too much memory)
            with torch.no_grad():
                # run model
                batch_embeds = self.forward(
                    input_ids=input_ids.to(device), 
                    input_lengths=input_lengths.to(device), 
                    step="train", 
                    return_only_embed=True,
                )

            embeddings.extend(batch_embeds)
            gold_labels.extend(torch.argmax(labels, dim=1))
            
        embeddings = torch.stack(embeddings)
        gold_labels = torch.Tensor(gold_labels) # [0, 1, 1, 4]

        for label_idx in range(len(self.label_encoder)): # label_idx = 0
            # create mask for label
            mask = gold_labels == label_idx # mask = [True, False, False, False]
            # embeddings mean for the label
            label_embedding = torch.mean(embeddings[mask], dim=0)

            # fill token_type_embeddings matrix
            #self.transformer.embeddings.token_type_embeddings.weight[label_idx].data.copy_(label_embedding.clone())
            self.sentiment_embeddings.weight[label_idx].data.copy_(label_embedding.clone())
            #embeddings_mean.append(label_embedding)

        #self.sentiment_embeddings = nn.Embedding.from_pretrained(torch.stack(embeddings_mean), freeze=False)
        
        # put model back in train mode
        self.transformer.train()

    def set_classification_head(self):
        in_features = self.encoder_features

        if self.config.pooling == "cls+avg":
            in_features *= 2

        in_features *= self.config.nr_layers_pooling

        if self.config.retrieval_augmentation:
            in_features += self.config.sentiment_embeddings_size

        if self.config.classif_setup == "linear":
            
            return nn.Linear(
                in_features=in_features, 
                out_features=len(self.label_encoder)
            )

        elif self.config.classif_setup == "linear + relu + linear":
            return nn.Sequential(
                nn.Linear(
                    in_features=in_features,
                    out_features=(self.encoder_features * self.config.nr_layers_pooling) // 2,
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=(self.encoder_features * self.config.nr_layers_pooling) // 2,
                    out_features=len(self.label_encoder),
                )
            )
        
        else:
            raise Exception(f"Unrecognized classification head setup {self.haprams.classif_head}")

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            click.secho("-- Encoder model fine-tuning", fg="yellow")
            for param in self.transformer.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.transformer.parameters():
            param.requires_grad = False
        self._frozen = True

    def layerwise_lr(self, lr: float, decay: float) -> list:
        """ Separates layer parameters and sets the corresponding learning rate to each layer.

        :param lr: Initial Learning rate.
        :param decay: Decay value.

        :return: List with grouped model parameters with layer-wise decaying learning rate
        """
        opt_parameters = [
            {
                "params": self.transformer.embeddings.parameters(),
                "lr": lr * decay ** (self.num_layers),
            }
        ]
        opt_parameters += [
            {
                "params": self.transformer.encoder.layer[l].parameters(),
                "lr": lr * decay ** (self.num_layers - 1 - l),
            }
            for l in range(self.num_layers - 1)
        ]
        return opt_parameters
    
    # Pytorch Lightning Method
    def configure_optimizers(self):
        layer_parameters = self.layerwise_lr(
            self.config.encoder_learning_rate, self.config.layerwise_decay
        )
        head_parameters = [
            {
                "params": self.classification_head.parameters(),
                "lr": self.config.learning_rate,
            }
        ]
        if self.config.retrieval_augmentation:
            sentiment_parameters = [
                {
                    "params": self.sentiment_embeddings.parameters(),
                    "lr": self.config.learning_rate,
                }
            ]
            if self.config.init_sentiment_embeddings and self.config.sentiment_embeddings_size != 1024:
                redimension_parameters = [
                    {
                        "params": self.redimension_embeddings.parameters(),
                        "lr": self.config.learning_rate,
                    }
                ]
            if self.config.sentiment_lambda:
                lambda_parameters = [
                    {
                        "params": self.sentiment_lambda.parameters(),
                        "lr": self.config.learning_rate
                    }
                ]

        parameters = layer_parameters + head_parameters  

        if self.config.retrieval_augmentation: 
            parameters += sentiment_parameters

            if self.config.init_sentiment_embeddings and self.config.sentiment_embeddings_size != 1024:
                parameters += redimension_parameters

            if self.config.sentiment_lambda:
                parameters += lambda_parameters

        optimizer = AdamW(
            parameters,
            lr=self.config.learning_rate,
            correct_bias=True,
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def forward(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        labels_nn: torch.Tensor = None,
        step="train",
        return_only_embed=False,
        **kwargs,
    ) -> torch.Tensor:
        # Reduce unnecessary padding.
        input_ids = input_ids[:, : input_lengths.max()]

        mask = lengths_to_mask(input_lengths, device=input_ids.device)
        
        # Run model
        if self.config.retrieval_augmentation and not return_only_embed:
            try:
                sentiment_embeds = self.sentiment_embeddings(labels_nn)
            except:
                import pdb; pdb.set_trace()
            
            if self.config.init_sentiment_embeddings and self.config.sentiment_embeddings_size != 1024:
                sentiment_embeds = self.redimension_embeddings(sentiment_embeds)

            if self.config.sentiment_lambda:
                sent_lambda = self.sentiment_lambda(labels_nn)
                sentiment_embeds *= sent_lambda

            # nn_sentiment_ids = labels_nn.view(-1, 1).repeat(1, input_ids.shape[1]).to(input_ids.device)
            # inputs_embeds = self.transformer.embeddings.word_embeddings(input_ids)
            # sentiment_embeds = self.sentiment_embeddings(nn_sentiment_ids)
            #inputs_embeds = inputs_embeds + sentiment_embeds
            #output = self.transformer(inputs_embeds=inputs_embeds, attention_mask=mask, output_hidden_states=True)
        
        output = self.transformer(
            input_ids=input_ids, 
            attention_mask=mask,
            output_hidden_states=True,
        )

        if len(output) == 3:
            last_hidden_state = output['last_hidden_state']
            pooler_output = output['pooler_output']
            word_embeddings = output['hidden_states']
        elif len(output) == 2:
            last_hidden_state = output['last_hidden_state']
            word_embeddings = output['hidden_states']
        else:
            raise Exception(f"Can't unpack values {self.haprams.pretrained_model}")
        
        if return_only_embed:
            return average_pooling(input_ids, last_hidden_state, mask, self.tokenizer.pad_index)
        
        # Pooling Layer
        sentemb = self.apply_pooling(input_ids, last_hidden_state, word_embeddings, mask)

        sentemb = self.dropout(sentemb)

        # Classify
        if self.config.retrieval_augmentation:
            return self.classification_head(torch.cat((sentemb, sentiment_embeds), dim=1))
        return self.classification_head(sentemb)

    def apply_pooling(
        self, 
        tokens: torch.Tensor, 
        last_hidden_state: torch.Tensor, 
        embeddings: torch.Tensor, 
        mask: torch.Tensor, 
    ) -> torch.Tensor:
        """ Gets a sentence embedding by applying a pooling technique to the word-level embeddings.
        
        :param tokens: Tokenized sentences [batch x seq_length].
        :param last_hidden_state: Word embeddings of the last layer [batch x seq_length x hidden_size].
        :param embeddings: Word embeddings of every layer of the model [nr_layers x batch x seq_length x hidden_size]
        :param mask: Mask that indicates padding tokens [batch x seq_length].

        :return: Sentence embeddings [batch x hidden_size].
        """
        if self.config.pooling == "max":
            sentemb = max_pooling(tokens, last_hidden_state, self.tokenizer.pad_index)

        elif self.config.pooling == "avg":
            sentemb = average_pooling(
                tokens, last_hidden_state, mask, self.tokenizer.pad_index
            )

        elif self.config.pooling == "cls":
            sentemb = last_hidden_state[:, 0, :]

        elif self.config.pooling == "cls+avg":
            cls_sentemb = last_hidden_state[:, 0, :]
            avg_sentemb = average_pooling(
                tokens, last_hidden_state, mask, self.tokenizer.pad_index
            )
            sentemb = torch.cat((cls_sentemb, avg_sentemb), dim=1)

        elif self.config.pooling == "concat":
            word_embeddings = torch.stack(embeddings)[-self.config.nr_layers_pooling:]
            sentemb = torch.cat([layer[:,0,:] for layer in word_embeddings], dim=1)

        elif self.config.pooling == "bilstm":
            word_embeddings = torch.stack(embeddings)[-self.config.nr_layers_pooling:]
            word_embeddings_concat = torch.cat([layer for layer in word_embeddings], dim=2)
            sentemb, _ = self.biLSTM(word_embeddings_concat)
            # we only want one hidden state of the biLSTM
            sentemb = sentemb[:, 0, :]

        elif self.config.pooling == "clsavg":
            word_embeddings = torch.stack(embeddings)
            cls_stack = torch.stack([layer[:,0,:] for layer in word_embeddings])
            sentemb = torch.mean(cls_stack, dim=0)

        else:
            raise Exception("Invalid pooling technique.")

        return sentemb


    def on_train_start(self):
        if self.config.retrieval_augmentation and self.config.init_sentiment_embeddings:
            self.init_sentiment_representations(self.train_dataloader())

    # Pytorch Lightning Method
    def training_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        if len(batch) == 4:
            input_ids, input_lengths, labels, labels_nn = batch
            logits = self.forward(
                input_ids=input_ids, 
                input_lengths=input_lengths, 
                labels_nn=labels_nn, 
                step="train"
            )
        else:
            input_ids, input_lengths, labels = batch
            logits = self.forward(
                input_ids=input_ids, 
                input_lengths=input_lengths, 
                step="train"
            )

        loss_value = self.loss(logits, labels)

        if (
            self.nr_frozen_epochs < 1.0
            and self.nr_frozen_epochs > 0.0
            and batch_nb > self.epoch_total_steps * self.nr_frozen_epochs
        ):
            self.unfreeze_encoder()
            self._frozen = False
        #predictions = torch.argmax(logits, dim=1)
        
        
        # can also return just a scalar instead of a dict (return loss_val)
        return {"loss": loss_value, "log": {"train_loss": loss_value}}

    # Pytorch Lightning Method
    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        if len(batch) == 4:
            input_ids, input_lengths, labels, labels_nn = batch
            logits = self.forward(
                input_ids=input_ids, 
                input_lengths=input_lengths, 
                labels_nn=labels_nn, 
                step="valid"
            )
        else:
            input_ids, input_lengths, labels = batch
            logits = self.forward(
                input_ids=input_ids, 
                input_lengths=input_lengths, 
                step="valid"
            )
        loss_value = self.loss(logits, labels)

        if self.config.dataset == "goemotions" or self.config.dataset == "ekman":
            # Turn logits into probabilities
            predictions = torch.sigmoid(logits)

            # Turn probabilities into binary predictions
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0

        else:
            predictions = torch.argmax(logits, dim=1)
        
        self.log("loss", loss_value)
        return {"val_loss": loss_value, "predictions": predictions, "labels": labels}

    # Pytorch Lightning Method
    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        
        predictions = torch.cat([o["predictions"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)
        # Computes Precision Recall and F1 for all classes
        precision_scores, recall_scores, f1_scores = [], [], []
        y_hat_predictions, y_labels = [], []
        
        # multi class classification
        if self.config.dataset == "goemotions":
            for _, index in self.label_encoder.items():
                y_hat = predictions[:, index].cpu().numpy()
                y = labels[:, index].cpu().numpy()

                precision = precision_score(y, y_hat, zero_division=0)
                recall = recall_score(y, y_hat, zero_division=0)
                f1 = (
                    0
                    if (precision + recall) == 0
                    else (2 * (precision * recall) / (precision + recall))
                )
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

                y_hat_predictions.extend(y_hat)
                y_labels.extend(y)

            # We will log the macro and micro-averaged metrics:
            metrics = {
                "macro-precision": torch.tensor(sum(precision_scores) / len(precision_scores)),
                "macro-recall": torch.tensor(sum(recall_scores) / len(recall_scores)),
                "macro-f1": torch.tensor(sum(f1_scores) / len(f1_scores)),
                "micro-precision": torch.tensor(precision_score(y_labels, y_hat_predictions, average='micro', zero_division=0)),
                "micro-recall": torch.tensor(recall_score(y_labels, y_hat_predictions, average='micro', zero_division=0)),
                "micro-f1": torch.tensor(f1_score(y_labels, y_hat_predictions, average='micro', zero_division=0)),
            }

        else:
            y_hat_predictions = predictions.cpu().numpy()
            y_labels = torch.argmax(labels, dim=1).cpu().numpy()


            # We will log the macro and micro-averaged metrics:
            metrics = {
                "macro-precision": torch.tensor(precision_score(y_labels, y_hat_predictions, average='macro', zero_division=0)),
                "macro-recall": torch.tensor(recall_score(y_labels, y_hat_predictions, average='macro', zero_division=0)),
                "macro-f1": torch.tensor(f1_score(y_labels, y_hat_predictions, average='macro', zero_division=0)),
                "micro-precision": torch.tensor(precision_score(y_labels, y_hat_predictions, average='micro', zero_division=0)),
                "micro-recall": torch.tensor(recall_score(y_labels, y_hat_predictions, average='micro', zero_division=0)),
                "micro-f1": torch.tensor(f1_score(y_labels, y_hat_predictions, average='micro', zero_division=0)),
            }
            self.log("macro-precision", metrics["macro-precision"].to(self.transformer.device), prog_bar=True)
            self.log("macro-recall", metrics["macro-recall"].to(self.transformer.device), prog_bar=True)
            self.log("macro-f1", metrics["macro-f1"].to(self.transformer.device), prog_bar=True)
            
        return {
            "progress_bar": metrics,
            "log": metrics,
        }

    # Pytorch Lightning Method
    def test_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """ Same as validation_step. """
        if len(batch) == 4:
            input_ids, input_lengths, labels, labels_nn = batch
            logits = self.forward(
                input_ids=input_ids, 
                input_lengths=input_lengths, 
                labels_nn=labels_nn, 
                step="test"
            )
        else:
            input_ids, input_lengths, labels = batch
            logits = self.forward(
                input_ids=input_ids, 
                input_lengths=input_lengths, 
                step="test"
            )
        loss_value = self.loss(logits, labels)

        if self.config.dataset == "goemotions" or self.config.dataset == "ekman":
            # Turn logits into probabilities
            predictions = torch.sigmoid(logits)

            # Turn probabilities into binary predictions
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0

        else:
            predictions = torch.argmax(logits, dim=1)
        
        return {"val_loss": loss_value, "predictions": predictions, "labels": labels, "logits": logits}

    # Pytorch Lightning Method
    def test_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """ Similar to the validation_step_end but computes precision, recall, f1 for each label."""
        predictions = torch.cat([o["predictions"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)
        loss_value = torch.stack([o["val_loss"] for o in outputs]).mean()
        logits = [o["logits"] for o in outputs]
        

        # Computes Precision Recall and F1 for all classes
        precision_scores, recall_scores, f1_scores = [], [], []

        y_hat_predictions, y_labels = [], []

        # multi class classification
        if self.config.dataset == "goemotions":
            for _, index in self.label_encoder.items():
                y_hat = predictions[:, index].cpu().numpy()
                y = labels[:, index].cpu().numpy()
                precision = precision_score(y, y_hat, zero_division=0)
                recall = recall_score(y, y_hat, zero_division=0)
                f1 = (
                    0
                    if (precision + recall) == 0
                    else (2 * (precision * recall) / (precision + recall))
                )
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

                # microaveraging
                y_hat_predictions.append(y_hat)
                y_labels.append(y)

            # We will log the macro and micro-averaged metrics:
            metrics = {
                "macro-precision": sum(precision_scores) / len(precision_scores),
                "macro-recall": sum(recall_scores) / len(recall_scores),
                "macro-f1": sum(f1_scores) / len(f1_scores),
                "micro-precision": torch.tensor(precision_score(y_labels, y_hat_predictions, average='micro', zero_division=0)),
                "micro-recall": torch.tensor(recall_score(y_labels, y_hat_predictions, average='micro', zero_division=0)),
                "micro-f1": torch.tensor(f1_score(y_labels, y_hat_predictions, average='micro', zero_division=0)),
            }
            for label, i in self.label_encoder.items():
                metrics[label + "-precision"] = precision_scores[i]
                metrics[label + "-recall"] = recall_scores[i]
                metrics[label + "-f1"] = f1_scores[i]
        # regular classification
        else:
            y_hat_predictions = predictions.cpu().numpy()
            y_labels = torch.argmax(labels, dim=1).cpu().numpy()
            
            # We will log the macro and micro-averaged metrics:
            metrics = {
                "macro-precision": torch.tensor(precision_score(y_labels, y_hat_predictions, average='macro', zero_division=0)),
                "macro-recall": torch.tensor(recall_score(y_labels, y_hat_predictions, average='macro', zero_division=0)),
                "macro-f1": torch.tensor(f1_score(y_labels, y_hat_predictions, average='macro', zero_division=0)),
                "micro-precision": torch.tensor(precision_score(y_labels, y_hat_predictions, average='micro', zero_division=0)),
                "micro-recall": torch.tensor(recall_score(y_labels, y_hat_predictions, average='micro', zero_division=0)),
                "micro-f1": torch.tensor(f1_score(y_labels, y_hat_predictions, average='micro', zero_division=0)),
                "weighted-f1": torch.tensor(f1_score(y_labels, y_hat_predictions, average='weighted', zero_division=0)),
                "accuracy": torch.tensor(accuracy_score(y_labels, y_hat_predictions)),
            }
            confusion_matrix = {}

            # metrics per class
            for label, i in self.label_encoder.items():
                metrics[label + "-precision"] = precision_score(y_labels, y_hat_predictions, average=None, zero_division=0)[i]
                metrics[label + "-recall"] = recall_score(y_labels, y_hat_predictions, average=None, zero_division=0)[i]
                metrics[label + "-f1"] = f1_score(y_labels, y_hat_predictions, average=None, zero_division=0)[i]
                confusion_matrix[label] = multilabel_confusion_matrix(y_labels, y_hat_predictions)[i]

            macro_f1_no_majority_class = 0
            micro_f1_no_majority_class = 0

            for label, i in self.label_encoder.items():
                if label != "neutral" and label != "no emotion":
                    macro_f1_no_majority_class += metrics[label + "-f1"]

            ep_confusion_matrix = {}

            for label, i in self.label_encoder.items():
                if label != "fear" and label != "non-neutral" and label != "surprise" and label != "disgust":
                    ep_confusion_matrix[label] = confusion_matrix[label]

            macro_f1_no_majority_class /= len(self.label_encoder)-1
            micro_f1_no_majority_class = micro_f1(confusion_matrix)

            micro_f1_emotion_push = micro_f1(ep_confusion_matrix)
            import pdb; pdb.set_trace()

        results_dict = error_analysis(labels, predictions, self.label_encoder)

        self.log('metrics', metrics)
        self.log('loss_value', loss_value)
        self.log('results_dict', results_dict)
        self.log("macro f1 without majority class", macro_f1_no_majority_class)
        self.log("micro f1 without majority class", micro_f1_no_majority_class)
        self.log("micro f1 emotionpush", micro_f1_emotion_push)

        return {
            "progress_bar": metrics,
            "log": metrics,
            "val_loss": loss_value,
            "results_dict": results_dict,
            "macro f1 without majority class": macro_f1_no_majority_class,
            "micro f1 without majority class": micro_f1_no_majority_class,
        }

    # Pytorch Lightning Method
    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    @classmethod
    def from_experiment(cls, experiment_folder: str):
        """Function that loads the model from an experiment folder.

        :param experiment_folder: Path to the experiment folder.

        :return: Pretrained model.
        """
        hparams_file = experiment_folder + "hparams.yaml"
        hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

        checkpoints = [
            file for file in os.listdir(experiment_folder + "checkpoints/") if file.endswith(".ckpt")
        ]
        
        checkpoint_path = experiment_folder + "checkpoints/" + checkpoints[-1]
        model = cls.load_from_checkpoint(
            checkpoint_path, hparams=Namespace(**hparams), strict=True
        )
        # Make sure model is in prediction mode
        model.eval()
        model.freeze()
        return model

    def predict(self, samples: List[str]) -> Dict[str, Any]:
        """ Predict function.

        :param samples: list with the texts we want to classify.

        :return: List with classified texts.
        """
        if self.training:
            self.eval()

        output = [{"text": sample} for sample in samples]
        # Create inputs
        input_ids = [self.tokenizer.encode(s) for s in samples]
        input_lengths = [len(ids) for ids in input_ids]
        samples = {"input_ids": input_ids, "input_lengths": input_lengths}
        # Pad inputs
        samples = DataModule.pad_dataset(samples)
        dataloader = DataLoader(
            TensorDataset(
                torch.tensor(samples["input_ids"]),
                torch.tensor(samples["input_lengths"]),
            ),
            batch_size=self.config.batch_size,
            num_workers=multiprocessing.cpu_count(),
        )

        i = 0
        with torch.no_grad():
            for input_ids, input_lengths in dataloader:
                logits = self.forward(input_ids, input_lengths)
                # Turn logits into probabilities
                probs = torch.sigmoid(logits)
                for j in range(probs.shape[0]):
                    label_probs = {}
                    for label, k in self.label_encoder.items():
                        label_probs[label] = probs[j][k].item()
                    output[i]["emotions"] = label_probs
                    i += 1
        return output
