from re import escape
import click
import faiss
import json
import os
import numpy as np
import torch

from model.utils import df_emotion_lines_push, df_daily_dialog, load_scenarioSA, load_json
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib.pyplot as plt

class TextRetrieval():
    """ Text Retrieval class that holds a faiss index to be used to retrieve nearest neighbors.
    
    """

    def __init__(self, sentiment_representation):
        self.dim_examples = None # max dimension of the embeddings. used to initialize the faiss index
        self.datastore = None # faiss index to be initialized
        
        self.representations = [] # vector with the representations of each sentiment. i-th position represents i-th sentiment
        self.sentiment_representation = sentiment_representation # type of sentiment representation to be used
        
        self.lookup_table = {} # hash table that matches sentence id to the correspondent label

        self.distances = []

    def _build_embeddings(self, data, model):
        """ Uses a model to create embeddings of data

        :param data: dictionary containing the sentences to be encoded
        :param model: model to encode the sentences
        """
        for i, example in tqdm(enumerate(data)):
            example['text'] = model.encode(example['text'], convert_to_tensor=True)

        return data

    def _build_embeddings_generation(self, data, model):
        """ Uses a model to create embeddings of data

        :param data: dictionary containing the sentences to be encoded
        :param model: model to encode the sentences
        """
        for i, example in tqdm(enumerate(data)):
            data[i]['bot'] = model.encode(example['bot'], convert_to_tensor=True)

        return data

    def _get_id_label(self, sent_id):
        return self.lookup_table[sent_id]

    def _add_examples(self, data):
        """ Adds examples to a faiss index.
        
        :param data: train data to populate index
        """
        for i, example in tqdm(enumerate(data)):
            if i == 0: # initialize faiss
                self.dim_examples = len(example['text'])
                self.datastore = faiss.IndexFlatL2(self.dim_examples)

            self.datastore.add(example['text'].reshape(1, -1).numpy())
            # lookup table stores the label corresponding to the sentence index
            self.lookup_table[i] = int(example['label'])

    def _build_nn_file(self, data, data_path, type="train"):
        """ Retrieves nearest neighbor of each sentence and builds json file with sentence id and 
            corresponding label of the nearest neighbor. The resulting json file is placed on 
            the data_path

        :param data: dictionary with encoded data
        :param data_path: path where the data is saved
        :param type: whether it is building the train/valid/test nearest neighbor sets
        """

        nn_file = {}

        for i, example in tqdm(enumerate(data)):
            nn_id = self._retrieve_nearest_neighbor(example['text'], type)
            label = self._get_id_label(int(nn_id))

            nn_file[i] = label

        # create and save json file
        with open(os.path.join(data_path, f"{type}_nn.json"), 'w') as outfile:
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

        self.distances_hist(type)

        self.distances = []

        #with open(os.path.join(data_path, f"{type}_nn_quorum_{quorum}.json"), 'w') as outfile:
        #    json.dump(nn_file, outfile)

    def _build_nn_file_quorum_generation(self, data, data_path, type="train", quorum=5):
        
        nn_file = {}

        for i, example in tqdm(enumerate(data)):
            nn_id = self._retrieve_nearest_neighbor_quorum(example['bot'], type, quorum)

            quorum_list = []
            for id in nn_id:
                quorum_list.append(self._get_id_label(int(id)))

            label = self.most_frequent(quorum_list)

            nn_file[i] = label

        with open(os.path.join(data_path, f"{type}_nn_quorum_{quorum}_generation_sentiment_sentences.json"), 'w') as outfile:
            json.dump(nn_file, outfile)
            

    def _build_nn_file_brute_force(self, data, data_train, data_path, type="train"):

        nn_file = {}

        max_label = 0

        data_stack = []
        data_train_stack = []
        
        for i, example in tqdm(enumerate(data)):
            data_stack.append(example['text'])

        if type == "train":
            data_train_stack = data_stack
        else:
            for i, example in enumerate(data_train): # data_train will always be our index
                data_train_stack.append(example['text'])
                
        cosine_score = util.pytorch_cos_sim(torch.stack(data_stack, dim=0), torch.stack(data_train_stack, dim=0))

        select_i = 1 if type == "train" else 0
        
        top = torch.topk(cosine_score, 2).indices

        for i, example in enumerate(top):
            nn_file[i] = data_train[top[i][select_i].item()]['label']

        # create and save json file
        with open(os.path.join(data_path, f"brute_force_{type}_nn.json"), 'w') as outfile:
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

        self.distances.append(D[0][1])

        return nearest_input[0]

        

    def build_dataset(self, dataset_path):
        """ Reads and saves dataset.

        :param dataset_path: path to the dataset
        """
        if not os.path.isdir(dataset_path):
            click.secho(f"{dataset_path} not found!", fg="red")

        with open(dataset_path + "labels.txt", "r") as fp:
            labels = [line.strip() for line in fp.readlines()]
            label_encoder = {labels[i]: i for i in range(len(labels))}

        if dataset_path == "../data/emotionpush/" or dataset_path == "../data/emotionlines/":
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

        elif dataset_path == "../data/scenariosa/":
            train, valid, test = load_scenarioSA(dataset_path + "InteractiveSentimentDataset/")  

        model = SentenceTransformer('paraphrase-distilroberta-base-v1')

        dataset = {
            "train": self._build_embeddings(train.to_dict("records"), model), 
            "valid": self._build_embeddings(valid.to_dict("records"), model), 
            "test": self._build_embeddings(test.to_dict("records"), model)
        }
        return dataset, model


    def faiss_method(self, dataset_path):
        dataset, model = self.build_dataset(dataset_path)

        self._add_examples(dataset['train'])

        self._build_nn_file(dataset['train'], dataset_path, type='train')
        self._build_nn_file(dataset['valid'], dataset_path, type='valid')
        self._build_nn_file(dataset['test'], dataset_path, type='test')


    def quorum_faiss_method(self, dataset_path, generation_data):
        dataset, model = self.build_dataset(dataset_path)
        
        self._add_examples(dataset['train'])

        if generation_data:
            gen_data = load_json(generation_data)
            gen_data = self._build_embeddings_generation(gen_data, model)
        
            self._build_nn_file_quorum_generation(gen_data, dataset_path, type='valid', quorum=3)
        else:
            self._build_nn_file_quorum(dataset['train'], dataset_path, type='train', quorum=3)
            self._build_nn_file_quorum(dataset['valid'], dataset_path, type='valid', quorum=3)
            self._build_nn_file_quorum(dataset['test'], dataset_path, type='test', quorum=3)


    def brute_force_method(self, dataset_path):
        dataset, model = self.build_dataset(dataset_path)
        self._build_nn_file_brute_force(dataset['train'], dataset['train'], dataset_path, type="train")
        self._build_nn_file_brute_force(dataset['valid'], dataset['train'], dataset_path, type="valid")
        self._build_nn_file_brute_force(dataset['test'], dataset['train'], dataset_path, type="test")

            
    def initialize_representations(self, nr_labels): # , embedding_size) return nn.Embedding
        """ Initializes sentiment representations

        :param nr_labels: number of existent labels in the dataset
        """
        if self.sentiment_representation == "simple":
            for i in range(nr_labels):
                self.representations.append([i] * 768)


    def distances_hist(self, type):
        fig, ax = plt.subplots()

        x = ["[0;10[", "[10;20[", "[20;30[", "[30;40[", "[40;50[", "[50;60[", "[60;70[", "[70;80[", "[80;90[", "[90;100]"]

        bins = list(np.arange(0, 110, step=10))
        counts, _ = np.histogram(self.distances, bins=bins)

        plt.bar(x, counts)

        plt.ylabel("Nr of examples")
        plt.xlabel("Euclidean distance")

        plt.xticks(rotation=45)
        
        if type == "valid":
            plt.title("Distance between the validation set and their nearest training examples")
        else:
            plt.title("Distance between the test set and their nearest training examples")
        
        plt.savefig(f"dailydialog_{type}_hist.png", bbox_inches='tight')
        import pdb; pdb.set_trace()