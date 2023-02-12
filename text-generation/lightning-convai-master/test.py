import json
import pandas as pd

from tqdm import tqdm

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

data_location = "../../data/emotionpush/train.json"

data_json = load_json(data_location)
#emotions = []

# data = pd.DataFrame(columns=['conv_id', 'speaker', 'utterance', 'emotion', 'annotation'])
data = pd.DataFrame(columns=['text', 'label', 'conv_id'])
conv_id = 0

last_sentence = {}

affected_turns = 0
turns = 0

for i, conv in tqdm(enumerate(data_json), desc="reading conversations"):
    for j, sentence in enumerate(conv):

        turns += 1

        if i == 0 and j == 0:
            last_sentence = sentence

        # if we change conversations, then we want to add the previous one to the dataframe
        elif j == 0:
            new_sentence = {}

            new_sentence['text'] = last_sentence['utterance']
            new_sentence['label'] = last_sentence['emotion']
            new_sentence['conv_id'] = conv_id
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
            data = data.append(new_sentence, ignore_index=True)

            last_sentence = sentence
        
    conv_id += 1

print(affected_turns)
print(turns)
import pdb; pdb.set_trace()