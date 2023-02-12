import glob
import json
import os
import re
import spacy

import numpy as np
import pandas as pd


from tqdm import tqdm

spacy_model = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser", "textcat"])

def remove_punctuation(query):
    query = re.sub('[?|\.|!|:|,|;|-|(|)|"|/]', " ", query)
    return query

"""Performs preprocessing techniques on input text

    args:
        text (string): text to be preprocessed
"""
def preprocessing(text, lower=True, punct=True):

    if lower: text = text.lower()

    if punct: text = remove_punctuation(text)

    return text



"""Loads json file.
    args:
        data_location (string): path to the json file
    output:
        data (dictionary)
"""
def load_json(data_location):
    with open(data_location, "r") as read_file:
        data = json.load(read_file)

    return data


"""Creates dataframe for EmotionLines/Push based on a dictionary
    args:
        data (dictionary): python dictionary with data
    output:
        df (pandas dataframe)
"""
def dict2dataframe_emotion_lines_push(data):
    emotions = []

    df = pd.DataFrame(columns=['conv_id', 'speaker', 'utterance', 'emotion', 'annotation'])
    conv_id = 0

    for conv in tqdm(data, desc="conversation"):
        for sentence in conv:
            new_sentence = sentence
            new_sentence['conv_id'] = conv_id
            df = df.append(new_sentence, ignore_index=True)
            
            if sentence['emotion'] not in emotions: emotions.append(sentence['emotion'])

        conv_id += 1

    return df, emotions


def text2dataframe_daily_dialog(data, labels):

    df = pd.DataFrame(columns=['conv_id', 'utterance', 'emotion'])
    conv_id = 0

    labels_lines = labels.readlines()

    # each line is a conversation
    for i, conversation in tqdm(enumerate(data), desc="conversation"):
        # sentences and sentences_labels will have the same length
        sentences = conversation.split("__eou__")
        sentences_labels = labels_lines[i].split(" ")

        # remove last element which is a \n
        sentences.pop()
        sentences_labels.pop() 

        new_sentence = {}

        for j, sentence in enumerate(sentences):
            new_sentence['utterance'] = sentence
            new_sentence['conv_id'] = conv_id
            new_sentence['emotion'] = sentences_labels[j]

            df = df.append(new_sentence, ignore_index=True)

        conv_id += 1
    
    return df

def text2dataframe_scenarioSA(
    data_location: str
    ):
    """
    Loads ScenarioSA dataset and splits it on train, validation and test sets
    :param data_location: path dataset folder
    """

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


if __name__ == "__main__": 
    data_location = "../data/ijcnlp_dailydialog/train/dialogues_train.txt"
    labels_location = "../data/ijcnlp_dailydialog/train/dialogues_emotion_train.txt"

    data_txt = open(data_location, "r")
    labels_txt = open(labels_location, "r")

    text2dataframe_daily_dialog(data_txt, labels_txt)