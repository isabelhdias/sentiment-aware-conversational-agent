import glob
import json
import os
import re
import torch

import nlpaug.augmenter.word as naw
import numpy as np
import pandas as pd

from tqdm import tqdm

emotions = ['neutral', 'non-neutral', 'joy', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

def average_pooling(
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    padding_index: int,
) -> torch.Tensor:
    """Average pooling function.

    :param tokens: Word ids [batch_size x seq_length]
    :param embeddings: Word embeddings [batch_size x seq_length x hidden_size]
    :param mask: Padding mask [batch_size x seq_length]
    :param padding_index: Padding value.
    """
    wordemb = mask_fill(0.0, tokens, embeddings, padding_index)
    sentemb = torch.sum(wordemb, 1)
    sum_mask = mask.unsqueeze(-1).expand(embeddings.size()).float().sum(1)
    return sentemb / sum_mask


def max_pooling(
    tokens: torch.Tensor, embeddings: torch.Tensor, padding_index: int
) -> torch.Tensor:
    """Max pooling function.

    :param tokens: Word ids [batch_size x seq_length]
    :param embeddings: Word embeddings [batch_size x seq_length x hidden_size]
    :param padding_index: Padding value.
    """
    return mask_fill(float("-inf"), tokens, embeddings, padding_index).max(dim=1)[0]


def mask_fill(
    fill_value: float,
    tokens: torch.tensor,
    embeddings: torch.tensor,
    padding_index: int,
) -> torch.tensor:
    """
    Function that masks embeddings representing padded elements.
    :param fill_value: the value to fill the embeddings belonging to padded tokens.
    :param tokens: The input sequences [bsz x seq_len].
    :param embeddings: word embeddings [bsz x seq_len x hiddens].
    :param padding_index: Index of the padding token.
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)


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
    #emotions = []

    # data = pd.DataFrame(columns=['conv_id', 'speaker', 'utterance', 'emotion', 'annotation'])
    data = pd.DataFrame(columns=['text', 'label', 'conv_id'])
    conv_id = 0

    for conv in tqdm(data_json, desc="reading conversations"):
        
        for sentence in conv:
            new_sentence = {}
            new_sentence['text'] = sentence['utterance']
            new_sentence['label'] = sentence['emotion']
            new_sentence['conv_id'] = conv_id
            data = data.append(new_sentence, ignore_index=True)
            
        conv_id += 1

    return data


def df_emotion_lines_push_edit(
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

        # if i == 100: break # DEBUG

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


def augment_dataset(
    dataset: dict,
    labels: dict,
    pretrained_model: str,
    ) -> dict: 
    """
    Augment the dataset with sentences based on existing ones.
    """
    print(f"Balancing dataset by augmentation using model {pretrained_model}.")

    augmented_dataset = pd.DataFrame()
    #aug = naw.ContextualWordEmbsAug(model_path=pretrained_model, action="substitute", device='cuda')
    aug = naw.SynonymAug(aug_src='wordnet')
    labels_count = dataset['label'].value_counts()

    print("Dataset balancing before")
    print(labels_count)

    # max label = [label index, label count]
    max_label = [labels_count.idxmax() , max(labels_count)]

    for label, index in tqdm(labels.items(), desc="Augmenting dataset"):
        label_slice = dataset.loc[dataset['label'] == labels[label]]
        augmented_dataset = augmented_dataset.append(label_slice, ignore_index=True)
        
        if int(index) == int(max_label[0]):
            continue
        
        slice_length = len(label_slice)
        replicas_per_sentence = (max_label[1] - slice_length) // slice_length

        for i, sentence in tqdm(label_slice.iterrows(), desc="replicating sentences"):

            original = sentence["text"]
            augmented_replicas = aug.augment(sentence["text"], n=replicas_per_sentence)

            if replicas_per_sentence == 1:
                if replicas_per_sentence == original:
                    continue
                else:
                    sentence['text'] = augmented_replicas
                    augmented_dataset = augmented_dataset.append(sentence, ignore_index=True)
            
            else:
                for replica in augmented_replicas:
                    if replica == original:
                        continue
                    else:
                        sentence['text'] = replica
                        augmented_dataset = augmented_dataset.append(sentence, ignore_index=True)

    labels_count = augmented_dataset['label'].value_counts()
    
    print("Dataset balancing after")
    print(labels_count)
    print("\n")

    return augmented_dataset


def error_analysis(
    labels: torch.tensor, 
    predictions: torch.tensor, 
    label_encoder: dict,
    ) -> dict:
    
    results_dict = {}
    labels = torch.argmax(labels, dim=1).cpu().numpy()
    
    for idx, label in enumerate(labels):
        if label not in results_dict:
            results_dict[label] = [0] * len(label_encoder)
        
        results_dict[label][predictions[idx]] += 1

    values, indices = torch.topk(predictions, 3)

    return results_dict


def micro_f1(
    confusion_matrices: dict,
    ) -> float:

    false_positives = 0
    false_negatives = 0
    true_positives = 0

    for label, conf_matrix in confusion_matrices.items():
        if label != "neutral" and label != "no emotion":
            false_positives += conf_matrix[0][1]
            false_negatives += conf_matrix[1][0]
            true_positives += conf_matrix[1][1]
    try:
        precision = true_positives / (true_positives + false_positives) if true_positives else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives else 0
    except:
        import pdb; pdb.set_trace()

    return (2 * precision * recall) / (precision + recall) if precision else 0

