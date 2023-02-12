import json

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, multilabel_confusion_matrix

from utils import df_emotion_lines_push, df_daily_dialog, micro_f1

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

    train.replace(label_encoder, inplace=True)
    valid.replace(label_encoder, inplace=True)
    test.replace(label_encoder, inplace=True)

    dataset = {
        "train": train.to_dict("records"),
        "valid": valid.to_dict("records"),
        "test": test.to_dict("records"),
    }

    return dataset, label_encoder

def read_nn(dataset_path):
    print("Reading nearest neighbor files")

    labels_nn = {}

    with open(dataset_path + "train_nn_quorum_3.json") as train_file:
        labels_nn['train'] = json.load(train_file)
    
    with open(dataset_path + "valid_nn_quorum_3.json") as valid_file:
        labels_nn['valid'] = json.load(valid_file)

    with open(dataset_path + "test_nn_quorum_3.json") as test_file:
        labels_nn['test'] = json.load(test_file)

    return labels_nn

def evaluate(dataset, labels_nn, label_encoder, set='train'):
    print(f"Evaluating {set} set")

    y_hat = []
    y = []

    for sent_id, label in labels_nn.items():
        y_hat.append(int(label))

    for sent in dataset:
        y.append(int(sent['label']))

    accuracy = accuracy_score(y_true=y, y_pred=y_hat)
    macro_f1 = f1_score(y_true=y, y_pred=y_hat, average='macro')
    weighted_f1 = f1_score(y_true=y, y_pred=y_hat, average='weighted')
    micr_f1 = f1_score(y_true=y, y_pred=y_hat, average='micro')

    print(f"Accuracy: {accuracy}")
    print(f"Macro-F1: {macro_f1}")
    print(f"Weighted-F1: {weighted_f1}")
    print(f"Micro-F1: {micr_f1}")

    metrics = {}

    for label, i in label_encoder.items():
        metrics[label + "-f1"] = f1_score(y, y_hat, average=None, zero_division=0)[i]

    print("\nMetrics per label")
    print(metrics)

    confusion_matrix = {}

    # metrics per class
    for label, i in label_encoder.items():
        metrics[label + "-precision"] = precision_score(y, y_hat, average=None, zero_division=0)[i]
        metrics[label + "-recall"] = recall_score(y, y_hat, average=None, zero_division=0)[i]
        metrics[label + "-f1"] = f1_score(y, y_hat, average=None, zero_division=0)[i]
        confusion_matrix[label] = multilabel_confusion_matrix(y, y_hat)[i]

    macro_f1_no_majority_class = 0
    micro_f1_no_majority_class = 0

    for label, i in label_encoder.items():
        if label != "neutral" and label != "no emotion":
            macro_f1_no_majority_class += metrics[label + "-f1"]

    macro_f1_no_majority_class /= len(label_encoder)-1
    micro_f1_no_majority_class = micro_f1(confusion_matrix)

    print(f"Macro-F1 no MC: {macro_f1_no_majority_class}")
    print(f"Micro-F1 no MC: {micro_f1_no_majority_class}")


def main():
    dataset, label_encoder = read_dataset("../../data/emotionpush/")
    nn = read_nn("../../data/emotionpush/")

    evaluate(dataset['train'], nn['train'], label_encoder, set='train')
    evaluate(dataset['valid'], nn['valid'], label_encoder, set='valid')
    evaluate(dataset['test'], nn['test'], label_encoder, set='test')

main()
