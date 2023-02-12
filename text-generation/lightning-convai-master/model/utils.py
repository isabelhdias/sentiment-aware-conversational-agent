import glob
import json
import os
import nltk
import re
import random

import numpy as np
import pandas as pd

from tqdm import tqdm

from model.ms_jaccard import evaluate_ms_jaccard
from model.tfidf_distance import evaluate_tfidf_distance

emotions = ['neutral', 'non-neutral', 'joy', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

def load_json(
    data_location: str,
    ) -> dict:
    """
    Loads json file.
    :param data_location: path to the json file
    """
    with open(data_location, "r") as read_file:
        data = json.load(read_file)

    return data


def df_emotion_lines_push(
    data_location: str
    ) -> dict:
    """
    Creates dictionary for EmotionLines/Push
    :param data_location: path to the json file
    """
    data_json = load_json(data_location)
    
    data = pd.DataFrame(columns=['text', 'label', 'conv_id', 'speaker'])

    conv_id = 0

    last_sentence = {}

    affected_turns = 0
    turns = 0

    for i, conv in tqdm(enumerate(data_json), desc="reading conversations"):
        for j, sentence in enumerate(conv):

            turns += 1

            if not sentence['utterance']:
                continue 

            elif i == 0 and j == 0:
                last_sentence = sentence

            # if we change conversations, then we want to add the previous one to the dataframe
            elif j == 0:
                new_sentence = {}

                new_sentence['text'] = last_sentence['utterance']
                new_sentence['label'] = last_sentence['emotion']
                new_sentence['conv_id'] = conv_id
                new_sentence['speaker'] = last_sentence['speaker']
                data = data.append(new_sentence, ignore_index=True)

                last_sentence = sentence
            
            # if the speakers are the same, join the sentences 
            elif sentence['speaker'] == last_sentence['speaker']:
                # if one of the labels is neutral we want to keep the non neutral one. 
                # if both are neutral or non-neutral, then the sentence is just labelled as neutral/non-neutral
                if last_sentence['emotion'] == 'neutral' or last_sentence['emotion'] == 'non-neutral': 
                    last_sentence['emotion'] = sentence['emotion']

                elif sentence['emotion'] == 'neutral' or sentence['emotion'] == 'non-neutral':
                    continue 

                # hopefully this will never happen
                elif last_sentence['emotion'] != 'neutral' and sentence['emotion'] != 'neutral' and last_sentence['emotion'] != sentence['emotion']:
                    index1 = emotions.index(last_sentence['emotion'])
                    index2 = emotions.index(sentence['emotion'])
                    
                    # choose the least common emotion
                    emotion_index = index1 if index1 > index2 else index2

                    last_sentence['emotion'] = emotions[emotion_index]

                last_sentence['utterance'] += " " + sentence['utterance']
            
            # if the speakers are not the same, then we want to add the last sentence to the dataframe
            # and save the new sentence.
            else:
                new_sentence = {}

                new_sentence['text'] = last_sentence['utterance']
                new_sentence['label'] = last_sentence['emotion']
                new_sentence['conv_id'] = conv_id
                new_sentence['speaker'] = last_sentence['speaker']
                data = data.append(new_sentence, ignore_index=True)

                last_sentence = sentence
            
        conv_id += 1

    return data


def df_daily_dialog(
    data_location: str,
    labels: str
    ) -> dict:
    """
    Loads dailydialog dataset
    :param data_location: path to txt file
    :param labels: path to labels file
    """

    data = open(data_location, "r")
    labels = open(labels, "r")

    df = pd.DataFrame(columns=['conv_id', 'text', 'label'])
    conv_id = 0

    labels_lines = labels.readlines()

    # each line is a conversation
    for i, conversation in tqdm(enumerate(data), desc="reading conversations"):
        # sentences and sentences_labels will have the same length
        sentences = conversation.split("__eou__")
        sentences_labels = labels_lines[i].split(" ")

        # remove last element which is a \n
        sentences.pop()
        sentences_labels.pop() 

        new_sentence = {}

        # if i == 5: break # DEBUG

        for j, sentence in enumerate(sentences):
            new_sentence['text'] = sentence
            new_sentence['conv_id'] = conv_id
            new_sentence['label'] = sentences_labels[j]

            df = df.append(new_sentence, ignore_index=True)

        conv_id += 1
    
    return df


def load_scenarioSA(
    data_location: str
    ):
    """
    Loads ScenarioSA dataset and splits it on train, validation and test sets
    :param data_location: path dataset folder
    """
    np.random.seed(42)

    df = pd.DataFrame(columns=['conv_id', 'speaker', 'text', 'label'])

    
    for file_path in tqdm(glob.glob(os.path.join(data_location, "*.txt")), desc=f"reading conversations"):

        conv_id = file_path.split("/")[-1].split(".txt")[0]
        
        conv = open(file_path, "r", errors="ignore")
        
        for utterance in conv:
            
            utterance = utterance.strip()

            # if it's empty string ignore
            if re.match('^(?![\s\S])', utterance): continue

            # ignore final evaluation of the conversation for now 
            if re.match('(\-1|1|0) (\-1|1|0)', utterance): continue
            
            new_sentence = {}
            
            new_sentence['conv_id'] = conv_id
            new_sentence['speaker'] = utterance[0]
            
            if utterance[-2] == "-":
                #new_sentence['label'] = utterance[-2] + utterance[-1]
                new_sentence['label'] = 0
                new_sentence['text'] = sentence = utterance[3:-2].strip()
            else:
                if re.match('\d', utterance[-1]):
                    new_sentence['label'] = int(utterance[-1]) + 1
                else:
                    new_sentence['label'] = 0
                new_sentence['text'] = utterance[3:-1].strip()
            
            df = df.append(new_sentence, ignore_index=True)
            
    train, valid, test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])            

    return train, valid, test


def dataframe2distractorsdataset(
    data, candidates_set, nr_dialogs, label_encoder,
):
    conv_id = 0
    dataset = []
    df_index = 0

    # select dialogue
    for conv_id in tqdm(range(nr_dialogs), desc="reading dialogs"):
        dialog = data.loc[data['conv_id'] == conv_id]
        dataset_dialog = {'utterances': []}
        history = []

        # for each sentence in the dialogue create example with: {candidates, history}
        for i in range(len(dialog)):
            sentence = dialog.loc[df_index]
            if not history: 
                history.append(sentence.loc['text'])
                df_index += 1

            elif not sentence.loc['text']:
                df_index += 1

            else:
                utterance = {'candidates': [], 'history': []}
                
                utterance['history'].extend(history.copy())
                
                label_encoder_keys = list(label_encoder.copy().keys())
                
                # remove gold label from list
                label_encoder_keys.remove(sentence.loc['label'])

                # shuffle so that different sentiments are selected as distractors (when nr candidates < nr sentiments)
                random.shuffle(label_encoder_keys)

                for label in label_encoder_keys:
                    # select slice of candidates set correspondent to label
                    label_df = candidates_set.loc[candidates_set['label'] == label]

                    # select one candidate at random
                    candidate = choose_candidate(label_df)

                    # append to candidates
                    utterance['candidates'].append(candidate)

                # last candidate is the true one
                utterance['candidates'].append(sentence.loc['text'])
                
                # add gold label
                utterance['gold_label'] = sentence['label']

                # add new utterance to dialog
                dataset_dialog['utterances'].append(utterance)
                
                # add sentence to history
                history.append(sentence.loc['text'])

                df_index += 1

        dataset.append(dataset_dialog)

    return dataset

def choose_candidate(df):
    candidate = df.sample()['text'].values[0]

    if candidate:
        return candidate

    return choose_candidate(df)

def lexical_overlap_metric(sentiment_corpus, sentences, sentiments):
    f = open("tfidf.txt", "w")

    sentiment_corpus = df_emotion_lines_push(sentiment_corpus)
    bot_replies = []

    sentences = pd.DataFrame.from_dict(sentences)

    results_jaccard = {}
    results_tfidf = {}

    # train sentences of a given sentiment vs. dev/test sentences of gold label
    for sentiment, _ in sentiments.items():
        sentiment_df = sentiment_corpus.loc[sentiment_corpus['label'] == sentiment]
        sentiment_texts = sentiment_df['text'].tolist()

        bot_replies_sentiment_df = sentences.loc[sentences['label'] == sentiment]
        bot_replies = bot_replies_sentiment_df['bot'].tolist()
   
        #result_jaccard = evaluate_ms_jaccard(sentiment_texts, bot_replies)
        
        #print(f"Result {sentiment}: {result_jaccard}\n")
        
        result_tfidf = evaluate_tfidf_distance(sentiment_texts, bot_replies)
        #print(f"Result {sentiment}: {result_tfidf}\n")

        #results_jaccard[sentiment] = result_jaccard
        results_tfidf[sentiment] = result_tfidf
        
        f.write(f"\n\n --- {sentiment} --- \n")
        for sent, _ in sentiments.items():
            sent_df = sentiment_corpus.loc[sentiment_corpus['label'] == sent]
            sent_texts = sent_df['text'].tolist()

            result_tfidf_vs = evaluate_tfidf_distance(sent_texts, bot_replies)
            
            f.write(f"Result {sent} vs. {sentiment}: {result_tfidf_vs['tfidf_distance']} \n")

    return results_tfidf #results_jaccard, results_tfidf
