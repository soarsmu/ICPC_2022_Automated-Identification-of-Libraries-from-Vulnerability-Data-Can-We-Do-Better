import pandas as pd
import numpy as np
import json
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split


cleaned_CPE_data_path = "dataset/CVE_CPE_cleaned.csv"
CVE_labels_data_path = "dataset/CVE_Labels_cleaned.csv"
DATASET_PATH = "dataset/final_dataset_merged_cleaned.csv"

# Read the cleaned dataset csv file and split the dataset into train and test data
def data_preparation(nrows=None):
    df_CPE = pd.read_csv(DATASET_PATH, usecols=["cve_id", "merged"], nrows=nrows)


    # Read column names from file
    cols = list(pd.read_csv(DATASET_PATH, nrows=1))
    df_labels = pd.read_csv(DATASET_PATH, usecols =[i for i in cols if i not in ["cve_id", "cleaned", "matchers", "merged"]], nrows=nrows)

    data = df_CPE.to_numpy()
    labels = df_labels.to_numpy()
    # Split dataset using skmultilearn (for multi-label classification)

    train, label_train, test, label_test = iterative_train_test_split(data, labels, test_size=0.25)
    # train, test, label_train, label_test = train_test_split(data, labels, test_size=0.25, random_state=0)
    # Save train and test data to file
    np.save("dataset/train_data.npy", train)
    np.save("dataset/test_data.npy", test)
    np.save("dataset/train_label.npy", label_train)
    np.save("dataset/test_label.npy", label_test)
    return train, label_train, test, label_test

def create_train_test_json(nrows=None):
    # use the splitted dataset to create the train and test json required for the fastxml algorithm
    train = np.load("dataset/train_data.npy", allow_pickle=True)
    test = np.load("dataset/test_data.npy", allow_pickle=True)
    # train = np.load("dataset/train_data_non_iterative.npy", allow_pickle=True)
    # test = np.load("dataset/test_data_non_iterative.npy", allow_pickle=True)
    df_labels = pd.read_csv(CVE_labels_data_path, usecols=["cve_id", "labels"], nrows=nrows)
    # with open("dataset/train_non_iterative.json", "w") as f:
    with open("dataset/train.json", "w") as f:
        for data in train:
            json_rep = {}
            json_rep["title"] = data[1].lstrip().rstrip()
            cve_id = data[0]
            cve_labels = df_labels[df_labels["cve_id"] == cve_id]["labels"].values.__str__()
            # Cleanup the label string from the cve_labels variable
            cve_labels = cve_labels.replace("[", "")
            cve_labels = cve_labels.replace("]", "")
            cve_labels = cve_labels.replace("'", "")
            cve_labels = cve_labels.replace('"', "")
            cve_labels = cve_labels.replace(" ", "")
            cve_labels = cve_labels.split(",")

            json_rep["tags"] = cve_labels
            json.dump(json_rep, f, ensure_ascii=False)
            f.write("\n")

    # with open("dataset/test_non_iterative.json", "w") as f:
    with open("dataset/test.json", "w") as f:
        for data in test:
            json_rep = {}
            json_rep["title"] = data[1].lstrip().rstrip()
            cve_id = data[0]
            cve_labels = df_labels[df_labels["cve_id"] == cve_id]["labels"].values.__str__()
            # Cleanup the label string from the cve_labels variable
            cve_labels = cve_labels.replace("[", "")
            cve_labels = cve_labels.replace("]", "")
            cve_labels = cve_labels.replace("'", "")
            cve_labels = cve_labels.replace('"', "")
            cve_labels = cve_labels.replace(" ", "")
            cve_labels = cve_labels.split(",")
            json_rep["tags"] = cve_labels
            json.dump(json_rep, f, ensure_ascii=False)
            f.write("\n")

def calculate_precision_recall(result_json_path: str, k: int):
    n_correct_prediction = 0
    n_actual_label = 0
    n_prediction_done = 0

    num_line = 0
    sum_precision = 0
    sum_recall = 0
    with open(result_json_path, "r", encoding="utf-8") as f:
        for line in f:
            local_correct_prediction = 0

            num_line += 1
            d = json.loads(line)
            actual_label = d["tags"]
            # get only the top k prediction
            prediction = d["predict"][0:k]
            # print(prediction)
            for pred in prediction:
                if pred[0] in actual_label:
                    n_correct_prediction += 1
                    local_correct_prediction += 1
                n_prediction_done += 1
            n_actual_label += len(actual_label)
            sum_precision += (local_correct_prediction / k)
            sum_recall += (local_correct_prediction / len(actual_label))
    print("Printing evaluation metrics @ " + k.__str__())
    print("Correct prediction = " + n_correct_prediction.__str__())
    print("Number of prediction done = " + n_prediction_done.__str__())
    print("Actual number of labels = " + n_actual_label.__str__())
    precision = (sum_precision / num_line)
    recall = (sum_recall / num_line)
    f1 = 2 * precision * recall / (precision + recall)
    print("Precision = " + precision.__str__())
    print("Recall = " + recall.__str__())
    print("F1 = " + f1.__str__())
    print()
    print()
#
# data_preparation()
# create_train_test_json()

# calculate_precision_recall("inference_result.json", 1)
# calculate_precision_recall("inference_result.json", 2)
# calculate_precision_recall("inference_result.json", 3)

def calculate_metrics_all(result_json_path):
    sum_recall_1 = 0
    sum_recall_2 = 0
    sum_recall_3 = 0
    sum_precision_1 = 0
    sum_precision_2 = 0
    sum_precision_3 = 0
    num_test_data = 0
    total_labels = 0
    with open(result_json_path, "r", encoding="utf-8") as f:
        for line in f:
            num_test_data += 1
            local_correct_prediction = 0

            d = json.loads(line)
            labels = d["tags"]
            # get only the top k prediction
            prediction = d["predict"][0:3]
            total_labels += len(labels)
            correct_prediction = 0
            # K = 1
            if prediction[0][0] in labels:
                correct_prediction += 1
            sum_precision_1 += (correct_prediction / 1)
            sum_recall_1 += (correct_prediction / len(labels))

            # K = 2
            if prediction[1][0] in labels:
                correct_prediction += 1
            sum_precision_2 += (correct_prediction / 2)
            sum_recall_2 += (correct_prediction / len(labels))

            # K = 3
            if prediction[2][0] in labels:
                correct_prediction += 1
            sum_precision_3 += (correct_prediction / 3)
            sum_recall_3 += (correct_prediction / len(labels))


        precision_1 = sum_precision_1 / num_test_data
        precision_2 = sum_precision_2 / num_test_data
        precision_3 = sum_precision_3 / num_test_data
        recall_1 = sum_recall_1 / num_test_data
        recall_2 = sum_recall_2 / num_test_data
        recall_3 = sum_recall_3 / num_test_data
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
        f1_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2)
        f1_3 = 2 * precision_3 * recall_3 / (precision_3 + recall_3)

        print("K = 1")
        print("P@1 = " + precision_1.__str__())
        print("R@1 = " + recall_1.__str__())
        print("F@1 = " + f1_1.__str__())

        print("K = 2")
        print("P@2 = " + precision_2.__str__())
        print("R@2 = " + recall_2.__str__())
        print("F@2 = " + f1_2.__str__())

        print("K = 3")
        print("P@3 = " + precision_3.__str__())
        print("R@3 = " + recall_3.__str__())
        print("F@3 = " + f1_3.__str__())
        print("TOTAL LABELS: " + total_labels.__str__())

calculate_metrics_all("inference_result.json")

# df = pd.read_csv("dataset/dataset_no_urls.csv")
# train_data = np.load("dataset/train_data.npy", allow_pickle=True)[:,0]
# test_data = np.load("dataset/test_data.npy", allow_pickle=True)[:,0]
# train_dataframe = df[df['cve_id'].isin(train_data)]
# test_dataframe = df[df['cve_id'].isin(test_data)]
# train_dataframe.to_csv("dataset/dataset_train.csv", index=False)
# test_dataframe.to_csv("dataset/dataset_test.csv", index=False)
# print(train_data)

# test_data = np.load("dataset/test_label.npy")