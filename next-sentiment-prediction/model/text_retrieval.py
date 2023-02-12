import click
import faiss
import json
import os
import numpy as np
import pandas as pd
import torch

from model.utils import df_emotion_lines_push, df_daily_dialog
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MODEL = 'paraphrase-distilroberta-base-v1'

class TextRetrieval():
    """ Text Retrieval class that holds a faiss index to be used to retrieve nearest neighbors.
    
    """

    def __init__(self, sentiment_representation):
        self.dim_examples = None # max dimension of the embeddings. used to initialize the faiss index
        self.datastore = None # faiss index to be initialized
        
        self.representations = [] # vector with the representations of each sentiment. i-th position represents i-th sentiment
        self.sentiment_representation = sentiment_representation # type of sentiment representation to be used
        
        self.lookup_table = {} # hash table that matches sentence id to the correspondent label

    def _build_embeddings(self, data, model):
        """ Uses a model to create embeddings of data

        :param data: dictionary containing the sentences to be encoded
        :param model: model to encode the sentences
        """
        for i, example in tqdm(enumerate(data), desc="Building embeddings"):
            example['text'] = model.encode(example['text'], convert_to_tensor=True)

        return data


    def _build_embeddings_concat(self, data, model):

        examples = []

        for i in tqdm(range(len(data) - 2), desc="Building embeddings"):
            if i != 0:
                samples = []
                conv_id = data[i]['conv_id']
                label = data[i]['label']
                for turn in range(2):
                    if data[i]['conv_id'] == data[i-(turn+1)]['conv_id']:   # and data[i]['conv_id'] == data[i+2]['conv_id']:
                        samples.insert(0, data[i-(turn+1)]['text']) #data[i]['text'] + " " + data[i+1]['text']
                        
                # encode sentences
                if samples:
                    sample_encoded = model.encode(' '.join(map(str, samples)) if len(samples) > 1 else samples, convert_to_tensor=True)

                    # create new example
                    new_example = {'text': sample_encoded, 'conv_id': conv_id, 'label': label}
                    examples.append(new_example)
       
        return examples


    def _get_id_label(self, sent_id):
        return self.lookup_table[sent_id]

    def _add_examples(self, data):
        """ Adds examples to a faiss index.
        
        :param data: train data to populate index
        """
        for i, example in tqdm(enumerate(data), desc="Adding examples to datastore"):
            
            if i == 0: # initialize faiss
                self.dim_examples = len(example['text'][0])
                self.datastore = faiss.IndexFlatL2(self.dim_examples)
            
            self.datastore.add(example['text'].reshape(1, -1).numpy())
            # lookup table stores the label corresponding to the sentence index
            self.lookup_table[i] = int(example['label'])


    def _build_nn_file(self, data, data_path, type="train", file_name=None):
        """ Retrieves nearest neighbor of each sentence and builds json file with sentence id and 
            corresponding label of the nearest neighbor. The resulting json file is placed on 
            the data_path

        :param data: dictionary with encoded data
        :param data_path: path where the data is saved
        :param type: whether it is building the train/valid/test nearest neighbor sets
        """

        nn_file = {}

        for i, example in tqdm(enumerate(data), desc=f"Building {type} file"):
            nn_id = self._retrieve_nearest_neighbor(example['text'], type)
            label = self._get_id_label(int(nn_id))

            nn_file[i] = label

        fn = f"{file_name}_{type}_nn.json" if file_name else f"{type}_nn.json"

        # create and save json file
        with open(os.path.join(data_path, fn), 'w') as outfile:
            json.dump(nn_file, outfile)

    
    def most_frequent(self, list):
        return max(set(list), key = list.count)

    def _build_nn_file_quorum(self, data, data_path, type="train", quorum=5):
        
        nn_file = {}

        for i, example in tqdm(enumerate(data)):
            nn_id = self._retrieve_nearest_neighbor_quorum(example['text'], type, quorum)

            quorum_list = []
            for id in nn_id:
                quorum_list.append(self._get_id_label(int(id)))

            label = self.most_frequent(quorum_list)

            nn_file[i] = label

        with open(os.path.join(data_path, f"nsp_{type}_nn_quorum_{quorum}.json"), 'w') as outfile:
            json.dump(nn_file, outfile)
        

    def _build_nn_file_brute_force(self, data, data_train, data_path, type="train"):

        nn_file = {}

        max_label = 0

        data_stack = []
        data_train_stack = []
        
        for i, example in tqdm(enumerate(data), desc=f"Building {type} file"):
            data_stack.append(example['text'])

        if type == "train":
            data_train_stack = data_stack
        else:
            for i, example in enumerate(data_train): # data_train will always be our index
                data_train_stack.append(example['text'])

        # results in a matrix queries x train sentences, meaning the similarity of each sentence in data with
        # every sentence in the train set.     
        cosine_score = util.pytorch_cos_sim(torch.stack(data_stack, dim=0), torch.stack(data_train_stack, dim=0))
        nn_index = 0
        for i in tqdm(range(len(cosine_score) - 2), desc="Finding best scores"): # queries
            best_average = 0
            best_label = 0
            
            if data[i]['conv_id'] == data[i+1]['conv_id'] and data[i]['conv_id'] == data[i+2]['conv_id']:
                try:
                    for j in range(len(cosine_score[i]) - 2): # train sentences
                        # we want to avoid matching the same sentence
                        if cosine_score[i][j].item() > 0.99999: continue
                        
                        # find best match j and j+1
                        if data_train[j]['conv_id'] == data_train[j+1]['conv_id'] and data_train[j]['conv_id'] == data_train[j+2]['conv_id']:
                            avg_aux = (cosine_score[i][j] + cosine_score[i+1][j+1]) / 2
                            if avg_aux > best_average:
                                best_average = avg_aux
                                best_label = data_train[j+2]['label']
                except: 
                    print("hi")
                    import pdb; pdb.set_trace()
                
                nn_file[nn_index] = best_label
                nn_index += 1
        # create and save json file
        with open(os.path.join(data_path, f"nsp_brute-force_{type}_nn.json"), 'w') as outfile:
            json.dump(nn_file, outfile)


    def _retrieve_nearest_neighbor(self, query_text, type="train"):
        """ Retrieves the nearest neighbor for an input query. If the input query belongs to the train
            set, this method returns the second best match because the input will be a sentence that 
            already exists in the index, so the best match will be itself. Otherwise, returns the best
            match.

        :param query_text: input query
        """
        k = 1

        if type == "train": k = 2
        
        D, I = self.datastore.search(query_text.reshape(1, -1).numpy(), k)
        nearest_input = I[:,k-1]

        return nearest_input[0]

    def _retrieve_nearest_neighbor_quorum(self, query_text, type="train", quorum=5):

        if type == "train": quorum += 1

        D, I = self.datastore.search(query_text.reshape(1, -1).numpy(), quorum)

        if type == "train": nearest_input = I[:, 1-quorum:]

        else: nearest_input = I

        return nearest_input[0]


    def build_dataset(self, dataset_path, method):
        """ Reads and saves dataset.

        :param dataset_path: path to the dataset
        """
        if not os.path.isdir(dataset_path):
            click.secho(f"{dataset_path} not found!", fg="red")

        with open(dataset_path + "labels.txt", "r") as fp:
            labels = [line.strip() for line in fp.readlines()]
            label_encoder = {labels[i]: i for i in range(len(labels))}

        if dataset_path == "../data/emotionpush/":
            train = df_emotion_lines_push(dataset_path + "train.json")
            valid = df_emotion_lines_push(dataset_path + "dev.json")
            test = df_emotion_lines_push(dataset_path + "test.json")

            train.replace(label_encoder, inplace=True)
            valid.replace(label_encoder, inplace=True)
            test.replace(label_encoder, inplace=True)

        elif dataset_path == "../data/dailydialog/":
            train = df_daily_dialog(
                dataset_path + "train/" + "dialogues_train.txt", dataset_path + "train/" + "dialogues_emotion_train.txt"
                )
            valid = df_daily_dialog(
                dataset_path + "validation/" + "dialogues_validation.txt", dataset_path + "validation/" + "dialogues_emotion_validation.txt"
                )
            test = df_daily_dialog(
                dataset_path + "test/" + "dialogues_test.txt", dataset_path + "test/" + "dialogues_emotion_test.txt"
                )
            
        model = SentenceTransformer(MODEL)

        if method == "concat":
            dataset = {
                "train": self._build_embeddings_concat(train.to_dict("records"), model), 
                "valid": self._build_embeddings_concat(valid.to_dict("records"), model), 
                "test": self._build_embeddings_concat(test.to_dict("records"), model)
            }
        else:
            dataset = {
                "train": self._build_embeddings(train.to_dict("records"), model), 
                "valid": self._build_embeddings(valid.to_dict("records"), model), 
                "test": self._build_embeddings(test.to_dict("records"), model)
            }
        return dataset


    def faiss_method(self, dataset_path, method):
        dataset = self.build_dataset(dataset_path, method)

        self._add_examples(dataset['train'])

        click.secho("Building train file", fg="yellow")
        self._build_nn_file(dataset['train'], dataset_path, type='train')
        click.secho("Building validation file", fg="yellow")
        self._build_nn_file(dataset['valid'], dataset_path, type='valid')
        click.secho("Building test file", fg="yellow")
        self._build_nn_file(dataset['test'], dataset_path, type='test')


    def brute_force_method(self, dataset_path, method):
        dataset = self.build_dataset(dataset_path, method)
        
        click.secho("Building train file", fg="yellow")
        self._build_nn_file_brute_force(dataset['train'], dataset['train'], dataset_path, type="train")
        click.secho("Building validation file", fg="yellow")
        self._build_nn_file_brute_force(dataset['valid'], dataset['train'], dataset_path, type="valid")
        click.secho("Building test file", fg="yellow")
        self._build_nn_file_brute_force(dataset['test'], dataset['train'], dataset_path, type="test")


    def faiss_sentences_concat_method(self, dataset_path, method):
        dataset = self.build_dataset(dataset_path, method)

        self._add_examples(dataset['train'])

        click.secho("Building train file", fg="yellow")
        self._build_nn_file(dataset['train'], dataset_path, type='train', file_name="nsp_concat")
        click.secho("Building validation file", fg="yellow")
        self._build_nn_file(dataset['valid'], dataset_path, type='valid', file_name="nsp_concat")
        click.secho("Building test file", fg="yellow")
        self._build_nn_file(dataset['test'], dataset_path, type='test', file_name="nsp_concat")


    def quorum_faiss_method(self, dataset_path, method):
        dataset = self.build_dataset(dataset_path, method)

        self._add_examples(dataset['train'])

        self._build_nn_file_quorum(dataset['train'], dataset_path, type='train', quorum=5)
        self._build_nn_file_quorum(dataset['valid'], dataset_path, type='valid', quorum=5)
        self._build_nn_file_quorum(dataset['test'], dataset_path, type='test', quorum=5)

            
    def initialize_representations(self, nr_labels): # , embedding_size) return nn.Embedding
        """ Initializes sentiment representations

        :param nr_labels: number of existent labels in the dataset
        """
        if self.sentiment_representation == "simple":
            for i in range(nr_labels):
                self.representations.append([i] * 768)