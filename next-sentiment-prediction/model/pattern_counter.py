import pandas as pd

from collections import Counter
from tqdm import tqdm

from utils import df_emotion_lines_push, df_daily_dialog, load_scenarioSA


def read_dataset(dataset_path):
    print("Reading dataset")
    
    with open(dataset_path + "labels.txt", "r") as fp:
        labels = [line.strip() for line in fp.readlines()]
        label_encoder = {labels[i]: i for i in range(len(labels))}
    
    if dataset_path == "../../data/emotionpush/" or dataset_path == "../../data/emotionlines/":
        train = df_emotion_lines_push(dataset_path + "train.json")
        valid = df_emotion_lines_push(dataset_path + "dev.json")
        test = df_emotion_lines_push(dataset_path + "test.json")
    
    elif dataset_path == "../../data/dailydialog/":
        train = df_daily_dialog(
                dataset_path + "train/" + "dialogues_train.txt", dataset_path + "train/" + "dialogues_emotion_train.txt"
                )
        valid = df_daily_dialog(
            dataset_path + "validation/" + "dialogues_validation.txt", dataset_path + "validation/" + "dialogues_emotion_validation.txt"
            )
        test = df_daily_dialog(
            dataset_path + "test/" + "dialogues_test.txt", dataset_path + "test/" + "dialogues_emotion_test.txt"
            )

    # train.replace(label_encoder, inplace=True)
    # valid.replace(label_encoder, inplace=True)
    # test.replace(label_encoder, inplace=True)

    dataset = {
        "train": train.to_dict("records"),
        "valid": valid.to_dict("records"),
        "test": test.to_dict("records"),
    }

    return dataset, label_encoder


def build_pattern_counter(dataset):
    pattern_counter = {}

    for i in tqdm(range(len(dataset) - 2), desc="reading dataset"):
        if dataset[i]['conv_id'] == dataset[i+1]['conv_id'] and dataset[i]['conv_id'] == dataset[i+2]['conv_id']:
            sentiment1 = dataset[i]['label']
            sentiment2 = dataset[i+1]['label']
            sentiment3 = dataset[i+2]['label']

            key = (sentiment1, sentiment2)
            
            if sentiment3 not in pattern_counter:
                pattern_counter[sentiment3] = []
                
            pattern_counter[sentiment3].append(key)

    for key, pairs in pattern_counter.items():
        sentiment_pattern_counter = dict(Counter(pairs))
        import pdb; pdb.set_trace()


    return pattern_counter


def main():
    dataset, label_encoder = read_dataset("../../data/emotionpush/")
    build_pattern_counter(dataset['train'])

main()