import os
import re
import json
import random
import numpy as np
from numpy.typing import NDArray
from nltk.stem.snowball import SnowballStemmer
import emoji
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Tuple, Union
from bs4 import BeautifulSoup as bs
import torch
import torch.nn as nn
import pandas as pd
import logging

if os.path.exists("main.log"):
    os.remove("main.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the overall logging level

# Create a FileHandler to log to a file
file_handler = logging.FileHandler("main.log")
file_handler.setLevel(logging.INFO)  # Set the logging level for the file
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Create a StreamHandler to log to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set the logging level for the console
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# Simple function to read in files as string
def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        logger.debug(f"read_file: {file_path}")
        return f.read()


# Simple function to list all files under a directory and it's sub-directories as a list of strings
def list_all_files(folder_path: str) -> List[str]:
    files = []
    logger.debug(f"list_all_files: {folder_path}")
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files


class DataProcessing:
    """Clean the tweet data and create a new folder cleaned_tweet with cleaned data"""

    """
    The init function of this class sets up the folder structure for cleaned versions of
    the tweets dataset that mirrors it's folder structure. It also performs some sanity checks
    to see if the dataset exists or not
    """

    def __init__(self, input_dir: str = "./tweet", output_dir: str = "./cleaned_tweet"):
        logger = logging.getLogger(__name__)
        # Check if the tweet folder exists
        if not (
            os.path.isdir(input_dir)
            and os.path.isdir(os.path.join(input_dir, "test", "negative"))
            and os.path.isdir(os.path.join(input_dir, "test", "positive"))
            and os.path.isdir(os.path.join(input_dir, "train", "negative"))
            and os.path.isdir(os.path.join(input_dir, "train", "positive"))
        ):
            raise Exception(
                "The tweet folder doesn't exist or is corrupted. Please check the folder and try again."
            )

        logger.info("Creating cleaned_tweet folder structure")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test", "negative"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test", "positive"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "train", "negative"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "train", "positive"), exist_ok=True)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.pos_vocab = set()
        self.neg_vocab = set()
        self.all_vocab = set()
        self.pos_vocab.add("<UNK>")
        self.neg_vocab.add("<UNK>")
        self.all_vocab.add("<UNK>")
        self.stemmer = SnowballStemmer("english")

    # If the words have more than one capital letter, keep it as is; otherwise, convert to lowercase
    def remain_capital_words(self, content: str) -> str:
        words = content.split()
        return " ".join(
            [
                word if sum(1 for ch in word if ch.isupper()) > 1 else word.lower()
                for word in words
            ]
        )

    # Main data cleaning function
    def data_cleaning(self, content: str) -> str:

        # remove HTML tags
        content = bs(content, "html.parser").get_text()

        # transform emoji
        content = emoji.emojize(content)

        # Removes capitalization
        content = self.remain_capital_words(content)

        # Removes punctuations
        content = re.sub(r"[^\w\s]", "", content)

        # Need the re to remove links like http, https, www
        content = re.sub(r"http\S+|www\S+|https\S+", "", content, flags=re.MULTILINE)

        # Stemming
        content = " ".join([self.stemmer.stem(word) for word in content.split()])

        return content.strip()

    """
    This function first performs data cleaning, then save all cleaned tweets into the cleaned_tweet folder,
    mirroring the tweet folder structure. Then, it creates word-to-id and id-to-word mappings and saves everything
    into vocab.json.
    """

    def save_cleaned_file(self):
        for subset in ["train", "test"]:
            logger.info(f"Cleaning data on {subset} dataset")
            all_files = list_all_files(os.path.join(self.input_dir, subset))
            for file_path in all_files:
                # Get the content of the sample and label from folder name
                content = self.data_cleaning(read_file(file_path))
                label = os.path.basename(os.path.dirname(file_path))

                # Tokenized clean data and put them into appropriate folder from label
                if subset == "train" and content != "":
                    tokens = content.split()
                    if label == "positive":
                        self.pos_vocab.update(tokens)
                        self.all_vocab.update(tokens)
                    else:
                        self.neg_vocab.update(tokens)
                        self.all_vocab.update(tokens)

                # Getting the relative path to input directory and mirroring it to output directory
                relpath = os.path.relpath(file_path, self.input_dir)
                output_path = os.path.join(self.output_dir, relpath)

                # Write cleaned tweet into cleaned_tweet according to label
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)

        # Create id mapping for positive and negative words for entire vocabulary
        pos_word2idx = {word: idx for idx, word in enumerate(self.pos_vocab)}
        neg_word2idx = {word: idx for idx, word in enumerate(self.neg_vocab)}
        logger.info("Created word-to-id mapping for entire vocabulary")

        # Write the entire vocabulary including the word-to-id and id-to-word mappings to vocab.json
        with open("vocab.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "positive": {
                        "vocabulary": list(self.pos_vocab),
                        "word2idx": pos_word2idx,
                        "idx2word": {idx: word for word, idx in pos_word2idx.items()},
                    },
                    "negative": {
                        "vocabulary": list(self.neg_vocab),
                        "word2idx": neg_word2idx,
                        "idx2word": {idx: word for word, idx in neg_word2idx.items()},
                    },
                    "all": {
                        "vocabulary": list(self.all_vocab),
                        "word2idx": {
                            word: idx for idx, word in enumerate(self.all_vocab)
                        },
                        "idx2word": {
                            idx: word for word, idx in enumerate(self.all_vocab)
                        },
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("Finished writting to vocab.json")


class TfIdfVector:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.vocabulary = set()
        self.update_vocabulary()
        self.word2idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vector_length = len(self.vocabulary)
        self.tfidf_table = np.zeros((len(documents), self.vector_length))

    def update_vocabulary(self):
        """Updated the vocabulary based on the documents"""
        for document in self.documents:
            tokens = document.split()
            self.vocabulary.update(tokens)
        self.vocabulary = sorted(self.vocabulary)

    def tf(self, document_tokens: List[str]) -> NDArray[np.float64]:
        """Generate TF vector for a document"""
        total_tokens = len(document_tokens)
        count = Counter(document_tokens)
        tf_dict = {word: freq / total_tokens for word, freq in count.items()}
        vector = np.zeros(self.vector_length)
        for key, value in tf_dict.items():
            idx = self.word2idx[key]
            vector[idx] = value
        return vector

    def idf(self, document_tokens: List[str]) -> NDArray[np.float64]:
        """Generate IDF vector for a document"""
        vector = np.zeros(self.vector_length)
        for word in document_tokens:
            idx = self.word2idx[word]
            docs_contain_word = sum(1 for doc in self.documents if word in doc)
            # Both add 1 to avoid making the idf became 0
            vector[idx] = (
                np.log((len(self.documents) + 1) / (docs_contain_word + 1)) + 1
            )
        return vector

    def transform(self, documents: List[str]):
        """Transform each document into a TF-IDF vector"""
        for idx, document in enumerate(documents):
            document_tokens = document.split()
            tf_vector = self.tf(document_tokens)
            idf_vector = self.idf(document_tokens)
            tf_idf_vector = tf_vector * idf_vector
            self.tfidf_table[idx] = tf_idf_vector / np.linalg.norm(tf_idf_vector)
        logger.info(
            f"Created tf_idf mapping for corpus with shape {self.tfidf_table.shape}"
        )


class FeedForwardNN:
    def __init__(self, layer_sizes: List[int], activations: List):
        self.layers = []
        if layer_sizes:
            assert len(layer_sizes) - 2 == len(activations), \
            "The number of activations must be two less than the number of layer sizes."

            for i in range(len(layer_sizes) - 1):
                input_size = layer_sizes[i]
                output_size = layer_sizes[i+1]

                # Create and add the dense layer (weights and biases)
                weight = np.random.rand(input_size, output_size) * 0.1
                bias = np.zeros((1, output_size)) * 0.1
                self.layers.append((weight, bias))

                # Add an activation function only for hidden layers
                if i < len(layer_sizes) - 2:
                    self.layers.append((activations[i], None))
            self.model_check()
        self.backward_loss = None

    def model_check(self):
        logger.info("Performing integrity check on model layers")
        for idx, layer in enumerate(self.layers):
            if idx % 2 == 0:
                if isinstance(layer[0], np.ndarray):
                    logger.info(f"Layer {idx} shape: {layer[0].shape}")
                    if idx < len(self.layers) - 1:
                        assert layer[0].shape[1] == self.layers[idx+2][0].shape[0],\
                            f"Output layer {idx} size({layer[0].shape[1]}) must match input layer {idx+2} size({self.layers[idx+2].shape[0]})"
                else:
                    raise Exception(f"Layer {idx} should contain a weight matrix, but instead was\n{layer}")
            if idx % 2 != 0:
                if isinstance(layer[0], str):
                    logger.info(f"Layer {idx} activation function {layer[0]}")
                else:
                    raise Exception(f"Layer {idx} should contain an activation function, but instead was\n{layer}")

    def layer(self, input_size: int, output_size: int):
        weight = np.random.rand(input_size, output_size) * 0.1
        bias = np.zeros((1, output_size)) * 0.1
        self.layers.append((weight, bias))

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivate_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def derivate_relu(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, x: List[np.ndarray]) -> np.ndarray:
        # for input features
        self.input = x
        self.info = []
        for obj in self.layers:
            if isinstance(obj[0], str):
                if obj[0] == "sigmoid":
                    store = [obj[0], x]
                    x = self.sigmoid(x)
                    # This will store the activation output for use in backpropagation
                    store.append(x)
                    self.info.append(store)

                elif obj[0] == "relu":
                    store = [obj[0], x]
                    x = self.relu(x)
                    # This will store the activation output for use in backpropagation
                    store.append(x)
                    self.info.append(store)

            else:
                weight, bias = obj
                store = [weight, bias, x]

                # This will output the z value before activation
                z = np.dot(x, weight) + bias
                store.append(z)

                # This will store the z value before activation for use in backpropagation
                self.info.append(store)
                x = z
        return x

    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray):
        diff = y_pred - y_true
        # dL/dy_pred to be used at the output layer (chain rule multiplies by sigmoid')
        self.backward_loss = (2.0 / y_pred.size) * diff
        return np.mean(diff * diff)

    def derivate_mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray):
        # Same as backward_loss above; provided if you call it directly elsewhere.
        return (2.0 / y_pred.size) * (y_pred - y_true)

    def binary_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray):
        # add eps to avoid log(0)
        eps = 1e-9
        return -np.mean(
            y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
        )

    def backward(self, learning_rate: float = 0.01):
        # Assuming MSE as loss function
        activation = None
        updated_layers = []

        # self.info is a list List[Tuple[str, np.ndarray]] or List[Tuple[np.ndarray, np.ndarray]]
        for layer in reversed(self.info):
            # Each layer store [weights or sigmoid, input, z]

            # This is the output layer
            if activation is None:
                weights = layer[0]
                bias = layer[1]
                z_input = layer[2]
                z_output = layer[3]
                activation_output = self.derivate_sigmoid(z_output)

                # calculate partial_c/ partial_w -> forward_output * partial_c/partial_z
                # self.backward_loss is derivate of loss function: constant

                # delta shape should be (1, output shape)
                delta = self.backward_loss * activation_output
                # z_input is our forward result, we use T to make it shape become (input_shape, 1)
                gradient_w = z_input.T @ delta

                # gradient shape will become (input_shape, output_shape). Same as our weight matrix
                weights -= learning_rate * gradient_w

                # gradient for bias
                gradient_b = np.sum(delta, axis=0, keepdims=True)
                bias -= learning_rate * gradient_b

                updated_layers.append([weights, bias])
                self.backward_loss = delta @ weights.T
                activation = "no activation"
                continue

            # hidden layers
            if isinstance(layer[0], str):
                # Activation layer
                if layer[0] == "sigmoid":
                    activation = "sigmoid"
                    updated_layers.append(("sigmoid", None))
                elif layer[0] == "relu":
                    activation = "relu"
                    updated_layers.append(("relu", None))
                else:
                    activation = "no activation"
            else:
                if activation:
                    weights = layer[0]
                    bias = layer[1]
                    z_input = layer[2]
                    z_output = layer[3]
                    if activation == "sigmoid":
                        activation_output = self.derivate_sigmoid(z_output)
                    elif activation == "relu":
                        activation_output = self.derivate_relu(z_output)
                    else:  # No activation
                        activation_output = 1

                    # delta shape should be (1, output shape)
                    delta = self.backward_loss * activation_output

                    gradient = z_input.T @ delta
                    weights -= learning_rate * gradient

                    gradient_b = np.sum(delta, axis=0, keepdims=True)
                    bias -= learning_rate * gradient_b

                    updated_layers.append([weights, bias])
                    self.backward_loss = delta @ weights.T
        self.layers = list(reversed(updated_layers))

    def predict(self, x: np.ndarray) -> np.ndarray:
        result = self.forward(x)
        return (self.sigmoid(result) > 0.5).astype(int)

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
    ):
        result = self.forward(x)
        y_pred = self.sigmoid(result)
        loss = self.mean_squared_error(y, y_pred)
        self.backward(learning_rate)

        return loss
    
    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            training_plan: List = [(500, 0.1), (800, 0.05), (1000, 0.01)],
            log_step: int = 10):
        logger.info(f"Starting training process with\n\
                    X: {x.shape}\n\
                    y: {y.shape}\n\
                    Learning rates: {training_plan}\n\
                    Log step: {log_step}")
        i = 0
        for epoch in range(1, training_plan[-1][0] + 1):
            if epoch > training_plan[i][0]:
                i += 1
            loss = self.train(x, y, training_plan[i][1])
            if epoch % log_step == 0:
                logger.info(f"Epoch: {epoch} - Loss: {loss} - Learning rate: {training_plan[i][1]}")


def main():
    # Did you git pull?
    if os.path.exists("cleaned_tweet"):
        logger.info("cleaned_tweet folder already exists. Skipping data cleaning.")
    else:
        processor = DataProcessing()
        processor.save_cleaned_file()
        logger.info("Data cleaning completed.")
    doc_list = []
    labels = []
    for lbl in ["positive", "negative"]:
        list_of_files = list_all_files(f"./cleaned_tweet/train/{lbl}")
        doc_list.extend([read_file(file) for file in list_of_files])
        labels.extend([1 if lbl == "positive" else 0] * len(list_of_files))

    tf_idf = TfIdfVector(doc_list)
    tf_idf.transform(doc_list)
    print(doc_list[0])
    print("TF-IDF vector shape for the first document:", tf_idf.tfidf_table[0].shape)
    print("TF-IDF vector for the first document:", tf_idf.tfidf_table[0])

    # Test FeedForwardNN
    logger.info("Test FFNN")
    input_shape = tf_idf.tfidf_table.shape[1]
    nn = FeedForwardNN(
        [input_shape, 20, 20, 1], 
        ["relu", "relu"]
    )

    y = np.array(labels).reshape(-1, 1)
    X = tf_idf.tfidf_table
    nn.fit(X, y)


def test():
    # Test whether neural network work
    nn = FeedForwardNN()
    nn.layer(4, 4)
    nn.layers.append(("sigmoid", None))
    nn.layer(4, 4)
    nn.layers.append(("sigmoid", None))
    nn.layer(4, 1)
    X_test = np.array([[0, 0, 1, 2], [0, 1, 1, 2], [1, 0, 1, 2], [1, 1, 1, 2]])
    y_test = np.array([[0], [0], [0], [1]])

    for epoch in range(1000):
        if epoch < 500:
            learning_rate = 0.1
        elif epoch < 800:
            learning_rate = 0.01
        else:
            learning_rate = 0.001
        loss = nn.train(X_test, y_test, learning_rate=learning_rate)
        print(f"Epoch {epoch}, Loss: {loss}")

    preds = nn.predict(X_test)
    print(preds.flatten())


if __name__ == "__main__":
    # test()
    main()
