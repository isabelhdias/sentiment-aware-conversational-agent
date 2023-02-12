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
import multiprocessing
import os
from pickle import STRING

import bert_score
import click
import pytorch_lightning as pl
import sacrebleu
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
from collections import defaultdict
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm

from model.data_module import DataModule
from model.gpt2 import PersonaGPT2
from pytorch_lightning import seed_everything
from trainer import TrainerConfig, build_trainer

from model.data_module_baseline import DataModuleBaseline
from model.gpt2_baseline import EmotionBaselineGPT2

from model.utils import df_emotion_lines_push, df_daily_dialog, load_json, dataframe2distractorsdataset, lexical_overlap_metric

from sentence_transformers import SentenceTransformer, util


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
    model_config = PersonaGPT2.ModelConfig(yaml_file)
    model = PersonaGPT2(model_config.namespace())
    data = DataModule(model.config, model.tokenizer)
    trainer.fit(model, data)


@cli.command(name="train-baseline")
@click.option(
    "--config",
    "-f",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configure YAML file",
)
def train_baseline(config: str) -> None:
    yaml_file = yaml.load(open(config).read(), Loader=yaml.FullLoader)
    # Build Trainer
    train_configs = TrainerConfig(yaml_file)
    seed_everything(train_configs.seed)
    trainer = build_trainer(train_configs.namespace())

    # Build Model
    model_config = EmotionBaselineGPT2.ModelConfig(yaml_file)
    model = EmotionBaselineGPT2(model_config.namespace())
    data = DataModuleBaseline(model.config, model.tokenizer)
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
    logging.disable(logging.WARNING)
    model = PersonaGPT2.from_experiment(experiment)
    click.secho("Hello my name is Emotion Chatbot. How are you today?", fg="yellow")

    sentiment_sentences = load_json("data/sentiment_sentences.json")
    for sentiment, sentences in sentiment_sentences.items():
        for i, sentence in enumerate(sentences):
            sentiment_sentences[sentiment][i] = model.tokenizer.encode(sentence)

    history = []
    while True:
        click.secho("Choose a sentiment: anger, disgust, fear, joy, neutral, non-neutral, sadness, surprise", fg="yellow")
        sentiment_text = input(">>> ")
        while not sentiment_text:
            print("Prompt should not be empty!")
            sentiment_text = input(">>> ")

        #while sentiment_text not in ["anger", "disgust", "fear", "joy", "happiness", "neutral", "non-neutral", "sadness", "surprise"]:
        #    print("Invalid sentiment!")
        #    sentiment_text = input(">>> ")

        sentiment_representation = sentiment_sentences[sentiment_text]

        click.secho("Talk", fg="yellow")
        raw_text = input(">>> ")
        while not raw_text:
            print("Prompt should not be empty!")
            raw_text = input(">>> ")

        history.append(model.tokenizer.encode(raw_text))
        bot_input = DataModule.build_input(
            tokenizer=model.tokenizer, sentiment_representation=sentiment_representation, history=history
        )

        history_ids = model.generate(
            input_ids=torch.LongTensor([bot_input["input_ids"]]),
            token_type_ids=torch.LongTensor([bot_input["token_type_ids"]]),
            max_length=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )
        bot_reply_ids = history_ids[:, len(bot_input["input_ids"]) :][0]
        bot_reply = model.tokenizer.decode(bot_reply_ids, skip_special_tokens=True)
        print("BOT: {}".format(bot_reply))
        #history.append(bot_reply_ids.tolist())


@cli.command(name="interact-baseline")
@click.option(
    "--experiment",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment folder containing the checkpoint we want to interact with.",
)
def interact_baseline(experiment: str) -> None:
    """Interactive mode command where we can have a conversation with a trained model
    that impersonates a Vegan that likes cooking and radical activities such as sky-diving.
    """
    logging.disable(logging.WARNING)
    model = EmotionBaselineGPT2.from_experiment(experiment)
    click.secho("Hello my name is Emotion Chatbot. How are you today?", fg="yellow")

    history = []
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print("Prompt should not be empty!")
            raw_text = input(">>> ")

        history.append(model.tokenizer.encode(raw_text))
        bot_input = DataModuleBaseline.build_input(
            tokenizer=model.tokenizer, history=history
        )

        history_ids = model.generate(
            input_ids=torch.LongTensor([bot_input["input_ids"]]),
            token_type_ids=torch.LongTensor([bot_input["token_type_ids"]]),
            max_length=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )
        bot_reply_ids = history_ids[:, len(bot_input["input_ids"]) :][0]
        bot_reply = model.tokenizer.decode(bot_reply_ids, skip_special_tokens=True)
        print("BOT: {}".format(bot_reply))
        history.append(bot_reply_ids.tolist())


@cli.command(name="test")
@click.option(
    "--experiment",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment folder containing the checkpoint we want to interact with.",
)
@click.option(
    "--test_set",
    type=click.Path(exists=True),
    required=True,
    help="Path to the json file containing the testset.",
)
@click.option(
    "--sentiment_accuracy_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the folder to save the sentiment accuracy json.",
)
@click.option(
    "--sentiment_corpus",
    type=click.Path(exists=True),
    required=True,
    help="Path to the sentiment corpus to be used in the lexical overlap metric",
)
@click.option(
    "--cuda/--cpu",
    default=True,
    help="Flag that either runs inference on cuda or in cpu.",
    show_default=True,
)
@click.option(
    "--seed",
    default=12,
    help="Seed value used during inference. This influences results only when using sampling.",
    type=int,
)
@click.option(
    "--sample/--search",
    default=True,
    help="Flag that either runs Nucleus-Sampling or Beam search.",
    show_default=True,
)
@click.option(
    "--top_p",
    default=0.9,
    help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)",
    type=float,
)
@click.option(
    "--temperature",
    default=0.9,
    help="Use temperature to decrease the sensitivity to low probability candidates when sampling.",
    type=float,
)
@click.option(
    "--num_beams",
    default=5,
    help="Number of beams during search.",
    type=int,
)
@click.option(
    "--to_json",
    default=False,
    help="Creates and exports model predictions to a JSON file.",
    show_default=True,
)
def test(
    experiment: str,
    test_set: str,
    sentiment_accuracy_path: str,
    sentiment_corpus: str,
    cuda: bool,
    seed: int,
    sample: bool,
    top_p: float,
    temperature: float,
    num_beams: int,
    to_json: str,
) -> None:
    """Testing function where a trained model is tested in its ability to rank candidate
    answers and produce replies.
    """
    logging.disable(logging.WARNING)
    model = PersonaGPT2.from_experiment(experiment)
    tokenizer = model.tokenizer
    seed_everything(seed)

    cuda = cuda and torch.cuda.is_available()
    if cuda:
        model.to("cuda")

    test_dataset = defaultdict(list)

    max_history = model.hparams.max_history

    model_name = experiment.split("/")[1]

    # Sentence Embedding Similarity Metric
    sentence_embedding_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    # Sentiment Accuracy Metric
    sentiment_accuracy = {'history': [], 'reply': [], 'gold_reply': [], 'gold_label': []}


    # Set to retrieve sentences from (train set usually)
    # Read words set file and tokenize it
    if model.config.sentiment_representation == "words-set":
        words_set = load_json(model.config.words_set_path)
        for sentiment, sent_set in words_set.items():
            for i, word in enumerate(sent_set):
                sent_set[i] = tokenizer.encode(word)

    if model.config.sentiment_representation == "random-sample":
        if sentiment_accuracy_path == "../../data/emotionpush/":
            sentences_per_sentiment_set = df_emotion_lines_push(sentiment_corpus)

        elif sentiment_accuracy_path == "../../data/dailydialog":
            sentences_per_sentiment_set = df_daily_dialog(sentiment_corpus)

        for i in range(len(sentences_per_sentiment_set)):
            sentences_per_sentiment_set.loc[i, "text"] = tokenizer.encode(sentences_per_sentiment_set.loc[i, "text"])

    if model.config.sentiment_representation == "sentiment-sentences":
        if sentiment_accuracy_path == "../../data/emotionpush/":
            sentiment_sentences = load_json("data/sentiment_sentences.json")
        elif sentiment_accuracy_path == "../../data/dailydialog":
            sentiment_sentences = load_json("data/sentiment_sentences_dailydialog.json")

        for sentiment, sentences in sentiment_sentences.items():
            for i, sentence in enumerate(sentences):
                sentiment_sentences[sentiment][i] = tokenizer.encode(sentence)

    with open(sentiment_accuracy_path + "labels.txt", "r") as fp:
        labels = [line.strip() for line in fp.readlines()]
        label_encoder = {labels[i]: i for i in range(len(labels))}

    dataset = df_emotion_lines_push(test_set)
    dataset = dataframe2distractorsdataset(dataset, dataset.copy(), 80, label_encoder)
    counter = 0
    replies, rankings, replies_save = [], [], []
    for i, dialog in tqdm(enumerate(dataset), desc="Scoring dialogs...", dynamic_ncols=True):
        
        for utterance in dialog["utterances"]:
            
            # 1) Prepares sentiment representation
            if model.config.sentiment_representation == "tag": 
                sentiment_representation = utterance["gold_label"]
            
            elif model.config.sentiment_representation == "words-set": 
                if words_set[utterance["gold_label"]]:
                    sentiment_representation = random.choices(words_set[utterance["gold_label"]], k=1)
    
                    # flatten list
                    sentiment_representation = [item for sublist in sentiment_representation for item in sublist]
                else: 
                    sentiment_representation = None
            
            elif model.config.sentiment_representation == "random-sample":
                # selects the slice of the dataframe correspondent to the gold label
                sentiment_df = sentences_per_sentiment_set.loc[sentences_per_sentiment_set['label'] == utterance["gold_label"]]
                # selects a random sentence from the slice
                sentiment_representation = sentiment_df.sample()['text'].values[0]
            
            elif model.config.sentiment_representation == "sentiment-sentences":
                sentiment_representation = sentiment_sentences[utterance["gold_label"]]

            else:
                sentiment_representation = None

            counter += 1
            # 2) Saves Ground-Truth
            ground_truth_reply = utterance["candidates"][-1]

            # 3) Prepares History
            history = utterance["history"][-(2 * model.hparams.max_history + 1) :]
            history_ids = [model.tokenizer.encode(h) for h in history]

            # 4) Rank Candidates in batch:
            batch = []
            for j, candidate in enumerate(utterance["candidates"]):
                candidate_ids = model.tokenizer.encode(candidate)
                instance = DataModule.build_input(
                    tokenizer=model.tokenizer,
                    sentiment_representation=sentiment_representation,
                    history=history_ids,
                    reply=candidate_ids,
                )
                batch.append(instance)

            # from list of dictionaries to dictionary of lists
            batch = {k: [d[k] for d in batch] for k in batch[0]}
            batch = DataModule.pad_dataset(batch)
            if cuda:
                batch = {k: torch.LongTensor(v).cuda() for k, v in batch.items()}
            else:
                batch = {k: torch.LongTensor(v) for k, v in batch.items()}

            mc_logits = model(**batch).mc_logits

            rankings.append(
                {
                    "sentiment_representation": sentiment_representation,
                    "history": history,
                    "candidates": utterance["candidates"],
                    "ranking": torch.topk(
                        mc_logits, len(utterance["candidates"])
                    ).indices.tolist(),
                }
            )

            # 5) Generates Reply
            bot_input = DataModule.build_input(
                tokenizer=model.tokenizer, history=history_ids, sentiment_representation=sentiment_representation,
            )
            # Nucleus Sampling
            if sample:
                history_ids = model.generate(
                    input_ids=torch.LongTensor([bot_input["input_ids"]]).cuda()
                    if cuda
                    else torch.LongTensor([bot_input["input_ids"]]),
                    token_type_ids=torch.LongTensor(
                        [bot_input["token_type_ids"]]
                    ).cuda()
                    if cuda
                    else torch.LongTensor([bot_input["token_type_ids"]]),
                    max_length=200,
                    do_sample=True,
                    top_p=top_p,
                    temperature=0.7,
                )
            # Beam Search
            else:
                history_ids = model.generate(
                    input_ids=torch.LongTensor([bot_input["input_ids"]]).cuda()
                    if cuda
                    else torch.LongTensor([bot_input["input_ids"]]),
                    token_type_ids=torch.LongTensor(
                        [bot_input["token_type_ids"]]
                    ).cuda()
                    if cuda
                    else torch.LongTensor([bot_input["token_type_ids"]]),
                    max_length=200,
                    num_beams=num_beams,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                )

            bot_reply_ids = history_ids[:, len(bot_input["input_ids"]) :][0]
            bot_reply = model.tokenizer.decode(bot_reply_ids, skip_special_tokens=True)
            
            for j, candidate in enumerate(utterance["candidates"][-model.config.num_candidates:]):
                # to calculate perplexity create examples with the generated sentence
                bot_reply_input = DataModule.build_input(
                    tokenizer=model.tokenizer,
                    history=[model.tokenizer.encode(h) for h in history],
                    sentiment_representation=sentiment_representation,
                    reply=model.tokenizer.encode(candidate),
                    lm_labels=bool(j == model.config.num_candidates - 1),
                )
                for input_name, input_array in bot_reply_input.items():
                    test_dataset[input_name].append(input_array)

            test_dataset["mc_labels"].append(model.config.num_candidates - 1)
            test_dataset["n_candidates"] = model.config.num_candidates

            # Sentence Embedding Similarity metric
            bot_reply_emb = sentence_embedding_model.encode(" ".join(wordpunct_tokenize(bot_reply.lower())))
            ground_truth_reply_emb = sentence_embedding_model.encode(ground_truth_reply)
            cos_sim = util.pytorch_cos_sim(bot_reply_emb, ground_truth_reply_emb)
            
            replies.append(
                {
                    "sentiment_representation": sentiment_representation,
                    "history": history,
                    "bot": " ".join(wordpunct_tokenize(bot_reply.lower())),
                    "human": ground_truth_reply,
                    "cosine_similarity": cos_sim[0], # SES
                    "label": utterance['gold_label'],
                }
            )

            replies_save.append(
                {
                    "history": history,
                    "bot": " ".join(wordpunct_tokenize(bot_reply.lower())),
                    "human": ground_truth_reply,
                    "cosine_similarity": cos_sim[0].tolist(), # SES
                    "label": utterance['gold_label']
                }
            )

            sentiment_accuracy['history'].append(history)
            sentiment_accuracy['reply'].append(" ".join(wordpunct_tokenize(bot_reply.lower())))
            sentiment_accuracy['gold_reply'].append(ground_truth_reply)
            sentiment_accuracy['gold_label'].append(utterance['gold_label'])

    # Save replies to a file
    with open('replies.json', 'w') as outfile:
        json.dump(replies_save, outfile)

    # Lexical overlap metric
    tfidf_scores = lexical_overlap_metric(sentiment_corpus, replies, label_encoder)

    #ngram = [1, 2, 3, 4, 5]
    #jaccard_scores_avg = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    #for sentiment, scores in jaccard_scores.items():
    #    for n in ngram:
    #        jaccard_scores_avg[n] += scores[f"ms_jaccard{n}"]

    #for n, scores in jaccard_scores_avg.items():
    #    jaccard_scores_avg[n] = scores / len(label_encoder)

    #click.secho(f"Jaccard scores average: {jaccard_scores_avg}", fg="yellow")
    click.secho(f"TF-IDF scores: {tfidf_scores}", fg="yellow")

    # 6) Runs Ranking Metrics
    hits_1, hits_5, hits_10 = [], [], []
    for ranks in rankings:
        hits_1.append((len(ranks["candidates"]) - 1) in ranks["ranking"][:1])
        hits_5.append((len(ranks["candidates"]) - 1) in ranks["ranking"][:5])
        hits_10.append((len(ranks["candidates"]) - 1) in ranks["ranking"][:10])

    click.secho("Hits@1: {}".format(sum(hits_1) / len(hits_1)), fg="yellow")
    click.secho("Hits@5: {}".format(sum(hits_5) / len(hits_5)), fg="yellow")
    click.secho("Hits@10: {}".format(sum(hits_10) / len(hits_10)), fg="yellow")

    # 7) Runs Generation Metrics
    refs = [[s["human"] for s in replies]]
    sys = [s["bot"] for s in replies]
    ses = [s["cosine_similarity"] for s in replies]
    ses = torch.Tensor(ses)

    bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True, tokenize="intl").score
    click.secho(f"BLEU: {bleu}", fg="blue")
    ter = sacrebleu.corpus_ter(sys, refs, no_punct=True).score
    click.secho(f"TER: {ter}", fg="blue")
    sentence_similarity = ses.mean()
    click.secho(f"Sentence Embedding Similarity: {sentence_similarity}", fg="blue")

    # Sentiment accuracy file
    with open(os.path.join(sentiment_accuracy_path, f"{model_name}_sentiment_accuracy.json"), 'w') as outfile:
        json.dump(sentiment_accuracy, outfile)

    # BERTScore returns precison, recall, f1.. we will use F1
    bertscore = float(
        bert_score.score(
            cands=sys,
            refs=refs[0],
            lang="en",
            verbose=False,
            nthreads=4,
        )[2].mean()
    )
    click.secho(f"BERTScore: {bertscore}", fg="blue")

    # 8) Saves results.
    if isinstance(to_json, str):
        data = {
            "results": {
                "BLEU": bleu,
                "TER": ter,
                "BERTScore": bertscore,
                "Hits@1": sum(hits_1) / len(hits_1),
                "Hits@5": sum(hits_5) / len(hits_5),
                "Hits@10": sum(hits_10) / len(hits_10),
            },
            "generation": replies,
            "ranking": rankings,
        }
        with open(to_json, "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        click.secho(f"Predictions saved in: {to_json}.", fg="yellow")
    
    # calculate perplexity
    test_dataset = DataModule.pad_dataset(test_dataset, padding=model.tokenizer.pad_index)

    MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
    tensor_dataset = []
    
    for input_name in MODEL_INPUTS:
        tensor = torch.tensor(test_dataset[input_name])

        # MC labels contain the labels within the batch!
        # Thats why we have to split the data according to those batches.
        if input_name != "mc_labels":
            tensor = tensor.view(
                (-1, test_dataset["n_candidates"]) + tensor.shape[1:]
            )

        tensor_dataset.append(tensor)
    
    test_dataset = TensorDataset(*tensor_dataset)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=model.config.batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
    )

    # Build a very simple trainer
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        deterministic=True,
        logger=False
    )

    trainer.test(model, test_dataloaders=test_dataloader)



@cli.command(name="test-baseline")
@click.option(
    "--experiment",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment folder containing the checkpoint we want to interact with.",
)
@click.option(
    "--test_set",
    type=click.Path(exists=True),
    required=True,
    help="Path to the json file containing the testset.",
)
@click.option(
    "--sentiment_accuracy_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the folder to save the sentiment accuracy json.",
)
@click.option(
    "--sentiment_corpus",
    type=click.Path(exists=True),
    required=True,
    help="Path to the sentiment corpus to be used in the lexical overlap metric",
)
@click.option(
    "--nsp_labels",
    type=click.STRING,
    required=False
)
@click.option(
    "--cuda/--cpu",
    default=True,
    help="Flag that either runs inference on cuda or in cpu.",
    show_default=True,
)
@click.option(
    "--seed",
    default=12,
    help="Seed value used during inference. This influences results only when using sampling.",
    type=int,
)
@click.option(
    "--sample/--search",
    default=True,
    help="Flag that either runs Nucleus-Sampling or Beam search.",
    show_default=True,
)
@click.option(
    "--top_p",
    default=0.9,
    help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)",
    type=float,
)
@click.option(
    "--temperature",
    default=0.9,
    help="Use temperature to decrease the sensitivity to low probability candidates when sampling.",
    type=float,
)
@click.option(
    "--num_beams",
    default=5,
    help="Number of beams during search.",
    type=int,
)
@click.option(
    "--to_json",
    default=False,
    help="Creates and exports model predictions to a JSON file.",
    show_default=True,
)
@click.option(
    "--bad_words",
    default=False,
    help="Whether or not to use bad_words on generate",
    show_default=True,
)
def test_baseline(
    experiment: str,
    test_set: str,
    sentiment_accuracy_path: str,
    sentiment_corpus: str,
    nsp_labels: str,
    cuda: bool,
    seed: int,
    sample: bool,
    top_p: float,
    temperature: float,
    num_beams: int,
    to_json: str,
    bad_words: bool,
) -> None:
    """Testing function where a trained model is tested in its ability to rank candidate
    answers and produce replies.
    """
    logging.disable(logging.WARNING)
    model = EmotionBaselineGPT2.from_experiment(experiment)
    tokenizer = model.tokenizer
    seed_everything(seed)

    cuda = cuda and torch.cuda.is_available()
    if cuda:
        model.to("cuda")

    test_dataset = defaultdict(list)

    max_history = model.hparams.max_history 

    model_name = experiment.split("/")[1]

    bad_words_ids = None

    if bad_words:
        bad_words_path = os.path.join(sentiment_accuracy_path, "bad_words.json")
        bad_words_dict = load_json(bad_words_path)

        for key, bad_words_list in bad_words_dict.items():
            for i, bad_word in enumerate(bad_words_list):
                bad_words_dict[key][i] = tokenizer.encode(bad_word)
    
    # Sentence Embedding Similarity Metric
    sentence_embedding_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    # Sentiment Accuracy Metric
    sentiment_accuracy = {'history': [], 'reply': [], 'gold_reply': [], 'gold_label': []}

    with open(sentiment_accuracy_path + "labels.txt", "r") as fp:
        labels = [line.strip() for line in fp.readlines()]
        label_encoder = {labels[i]: i for i in range(len(labels))}

    # Set to retrieve sentences from (train set usually)
    # Read words set file and tokenize it
    if model.config.sentiment_representation == "words-set":
        words_set = load_json(model.config.words_set_path)
        for sentiment, sent_set in words_set.items():
            for i, word in enumerate(sent_set):
                sent_set[i] = tokenizer.encode(word)

    if model.config.sentiment_representation == "random-sample":
        if sentiment_accuracy_path == "../../data/emotionpush/":
            sentences_per_sentiment_set = df_emotion_lines_push(sentiment_corpus)
        elif sentiment_accuracy_path == "../../data/dailydialog/":
            sentences_per_sentiment_set = df_daily_dialog(sentiment_corpus + "dialogues_train.txt", sentiment_corpus + "dialogues_emotion_train.txt")

        for i in range(len(sentences_per_sentiment_set)):
            sentences_per_sentiment_set.loc[i, "text"] = tokenizer.encode(sentences_per_sentiment_set.loc[i, "text"])

    if model.config.sentiment_representation == "sentiment-sentences":
        if sentiment_accuracy_path == "../../data/emotionpush/":
            sentiment_sentences = load_json("data/sentiment_sentences.json")
        elif sentiment_accuracy_path == "../../data/dailydialog/":
            sentiment_sentences = load_json("data/sentiment_sentences_dailydialog.json")

        for sentiment, sentences in sentiment_sentences.items():
            for i, sentence in enumerate(sentences):
                sentiment_sentences[sentiment][i] = tokenizer.encode(sentence)
                    
    
    with open(sentiment_accuracy_path + "labels.txt", "r") as fp:
        labels = [line.strip() for line in fp.readlines()]
        label_encoder_nsp = {i: labels[i] for i in range(len(labels))}
    
    if nsp_labels:
        with open(nsp_labels) as nsp_labels_file:
            labels_nsp = []
            for labels in nsp_labels_file:
                for label in labels:
                    if sentiment_accuracy_path == "../../data/emotionpush/":
                        labels_nsp.append(label_encoder_nsp[int(label)])
                    elif sentiment_accuracy_path == "../../data/dailydialog/":
                        labels_nsp.append(label)

    if sentiment_accuracy_path == "../../data/emotionpush/":
        dataset = df_emotion_lines_push(test_set).to_dict("records")
    elif test_set == "../../data/dailydialog/validation/":
        dataset = df_daily_dialog(test_set + "dialogues_validation.txt", test_set + "dialogues_emotion_validation.txt").to_dict("records")
    elif test_set == "../../data/dailydialog/test/":
        dataset = df_daily_dialog(test_set + "dialogues_test.txt", test_set + "dialogues_emotion_test.txt").to_dict("records")
    
    replies = []
    replies_save = []
    limit = len(dataset) - max_history 
    nsp_counter = 0
    
    for i, example in tqdm(enumerate(dataset), desc="Generating replies", total=len(dataset)):
        if i >= limit: break

        if i > max_history - 1:

            # prepare history
            history = []
            for turn in range(max_history):
                if example['conv_id'] == dataset[i-(turn+1)]['conv_id']:
                    history.append(dataset[i-(turn+1)]["text"])
            
            history_ids = [model.tokenizer.encode(h) for h in history]

            if not history:
                continue

            # 1) Prepares sentiment representation
            if model.config.sentiment_representation == "tag": 
                sentiment_representation = tokenizer.encode(example["label"])
            
            elif model.config.sentiment_representation == "words-set": 
                if words_set[example["label"]]:
                    sentiment_representation = random.choices(words_set[example["label"]], k=1)
    
                    # flatten list
                    sentiment_representation = [item for sublist in sentiment_representation for item in sublist]
                else: 
                    sentiment_representation = None
            
            elif model.config.sentiment_representation == "random-sample":
                # selects the slice of the dataframe correspondent to the gold label
                sentiment_df = sentences_per_sentiment_set.loc[sentences_per_sentiment_set['label'] == example["label"]]
                # selects a random sentence from the slice
                sentiment_representation = sentiment_df.sample()['text'].values[0]
            
            elif model.config.sentiment_representation == "sentiment-sentences":
                if nsp_labels:
                    sentiment_representation = sentiment_sentences[labels_nsp[nsp_counter]]
                    nsp_counter += 1
                else:
                    sentiment_representation = sentiment_sentences[example["label"]]
                    
            else:
                sentiment_representation = None

            # save ground truth
            ground_truth_reply = example['text']
            try:
                # generate reply
                bot_input = DataModuleBaseline.build_input(
                    tokenizer=model.tokenizer, history=history_ids, sentiment_representation=sentiment_representation,
                )
            except:
                import pdb; pdb.set_trace()

            bad_words_ids = []

            if bad_words:
                label_key = example['label']

                for key, bad_words in bad_words_dict.items():
                    if key != label_key:
                        bad_words_ids.extend(bad_words)

            if sample:
                history_ids = model.generate(
                    input_ids=torch.LongTensor([bot_input["input_ids"]]).cuda()
                    if cuda
                    else torch.LongTensor([bot_input["input_ids"]]),
                    token_type_ids=torch.LongTensor(
                        [bot_input["token_type_ids"]]
                    ).cuda()
                    if cuda
                    else torch.LongTensor([bot_input["token_type_ids"]]),
                    max_length=200,
                    do_sample=True,
                    top_p=top_p,
                    temperature=0.7,
                    output_scores=True,
                    #bad_words_ids=bad_words_ids,
                )
            # Beam Search
            else:
                history_ids = model.generate(
                    input_ids=torch.LongTensor([bot_input["input_ids"]]).cuda()
                    if cuda
                    else torch.LongTensor([bot_input["input_ids"]]),
                    token_type_ids=torch.LongTensor(
                        [bot_input["token_type_ids"]]
                    ).cuda()
                    if cuda
                    else torch.LongTensor([bot_input["token_type_ids"]]),
                    max_length=400,
                    num_beams=num_beams,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    output_scores=True,
                    #bad_words_ids=bad_words_ids,
                )
                

            bot_reply_ids = history_ids[:, len(bot_input["input_ids"]) :][0]
            bot_reply = model.tokenizer.decode(bot_reply_ids, skip_special_tokens=True)

            # to calculate perplexity
            bot_reply_input = DataModuleBaseline.build_input(
                tokenizer=model.tokenizer,
                history=[model.tokenizer.encode(h) for h in history],
                sentiment_representation=sentiment_representation,
                reply=model.tokenizer.encode(ground_truth_reply),
                lm_labels=True,
            )
            for input_name, input_array in bot_reply_input.items():
                test_dataset[input_name].append(input_array)

            # Sentence Embedding Similarity metric
            bot_reply_emb = sentence_embedding_model.encode(" ".join(wordpunct_tokenize(bot_reply.lower())))
            ground_truth_reply_emb = sentence_embedding_model.encode(ground_truth_reply)
            cos_sim = util.pytorch_cos_sim(bot_reply_emb, ground_truth_reply_emb)

            replies.append(
                {
                    "history": history,
                    "bot": " ".join(wordpunct_tokenize(bot_reply.lower())),
                    "human": ground_truth_reply,
                    "cosine_similarity": cos_sim[0], # SES
                    "label": example['label']
                }
            )
            
            replies_save.append(
                {
                    "history": history,
                    "bot": " ".join(wordpunct_tokenize(bot_reply.lower())),
                    "human": ground_truth_reply,
                    "cosine_similarity": cos_sim[0].tolist(), # SES
                    "label": example['label']
                }
            )

            sentiment_accuracy['history'].append(history)
            sentiment_accuracy['reply'].append(" ".join(wordpunct_tokenize(bot_reply.lower())))
            sentiment_accuracy['gold_reply'].append(ground_truth_reply)
            sentiment_accuracy['gold_label'].append(example['label'])
    
    
    # Save replies to a file
    with open('replies.json', 'w') as outfile:
        json.dump(replies_save, outfile)

    # Lexical overlap metric # TODO esta metrica para dailydialog
    tfidf_scores = lexical_overlap_metric(sentiment_corpus, replies, label_encoder)

    #ngram = [1, 2, 3, 4, 5]
    #jaccard_scores_avg = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    #for sentiment, scores in jaccard_scores.items():
    #    for n in ngram:
    #        jaccard_scores_avg[n] += scores[f"ms_jaccard{n}"]

    #for n, scores in jaccard_scores_avg.items():
    #    jaccard_scores_avg[n] = scores / len(label_encoder)

    #click.secho(f"Jaccard scores average: {jaccard_scores_avg}", fg="yellow")
    #click.secho(f"TF-IDF scores: {tfidf_scores}", fg="yellow")

    click.secho("Evaluating model", fg="yellow")

    # Generation Metrics
    refs = [[s["human"] for s in replies]]
    sys = [s["bot"] for s in replies]
    ses = [s["cosine_similarity"] for s in replies]
    ses = torch.Tensor(ses)
   
    bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True, tokenize="intl").score
    click.secho(f"BLEU: {bleu}", fg="blue")
    ter = sacrebleu.corpus_ter(sys, refs, no_punct=True).score
    click.secho(f"TER: {ter}", fg="blue")
    sentence_similarity = ses.mean()
    click.secho(f"Sentence Embedding Similarity: {sentence_similarity}", fg="blue")

    with open(os.path.join(sentiment_accuracy_path, f"{model_name}_sentiment_accuracy.json"), 'w') as outfile:
        json.dump(sentiment_accuracy, outfile)

    # BERTScore returns precison, recall, f1.. we will use F1
    bertscore = float(
        bert_score.score(
            cands=sys,
            refs=refs[0],
            lang="en",
            verbose=False,
            nthreads=4,
        )[2].mean()
    )
    click.secho(f"BERTScore: {bertscore}", fg="blue")

    # save results
    if isinstance(to_json, str):
        data = {
            "results": {
                "BLEU": bleu,
                "TER": ter,
                "BERTScore": bertscore,
            },
            "generation": replies,
        }
        with open(to_json, "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        click.secho(f"Predictions saved in: {to_json}.", fg="yellow")
    
    # calculate perplexity
    test_dataset = DataModuleBaseline.pad_dataset(test_dataset, padding=model.tokenizer.pad_index)

    MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
    tensor_dataset = []

    for input_name in MODEL_INPUTS:
        tensor = torch.tensor(test_dataset[input_name])
        tensor_dataset.append(tensor)

    test_dataset = TensorDataset(*tensor_dataset)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=model.config.batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
    )

    # Build a very simple trainer
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        deterministic=True,
        logger=False
    )

    trainer.test(model, test_dataloaders=test_dataloader)

if __name__ == "__main__":
    cli()
