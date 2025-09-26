import os
import re
import json
import random
import numpy as np
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


# Configure basic logging to a file
logging.basicConfig(
    filename="main.log",  # Name of the log file
    level=logging.INFO,  # Minimum logging level to capture (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format="%(asctime)s:%(funcName)s:%(levelname)s:%(message)s",  # Format of the log messages
)
logger = logging.getLogger(__name__)


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
            and os.path.isdir(os.path.join(output_dir, "test", "negative"))
            and os.path.isdir(os.path.join(output_dir, "test", "positive"))
            and os.path.isdir(os.path.join(output_dir, "train", "negative"))
            and os.path.isdir(os.path.join(output_dir, "train", "positive"))
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

    def update_vocabulary(self):
        """Updated the vocabulary based on the documents"""
        for document in self.documents:
            tokens = document.split()
            self.vocabulary.update(tokens)
        self.vocabulary = sorted(self.vocabulary)

    def tf(self, document_tokens: List[str]) -> np.array:
        """Generate TF vector for a document"""
        total_tokens = len(document_tokens)
        count = Counter(document_tokens)
        tf_dict = {word: freq / total_tokens for word, freq in count.items()}
        vector = np.zeros(self.vector_length)
        for key, value in tf_dict.items():
            idx = self.word2idx[key]
            vector[idx] = value
        return vector

    def idf(self, document_tokens: List[str]) -> np.array:
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

    def transform(self, document: str) -> np.array:
        """Transform the document into a TF-IDF vector"""
        document_tokens = document.split()
        tf_vector = self.tf(document_tokens)
        idf_vector = self.idf(document_tokens)
        tf_idf_vector = tf_vector * idf_vector
        tf_idf_vector = tf_idf_vector / np.linalg.norm(tf_idf_vector)
        return tf_idf_vector


def main():
    # Did you git pull?
    if os.path.exists("cleaned_tweet"):
        logger.info("cleaned_tweet folder already exists. Skipping data cleaning.")
    else:
        processor = DataProcessing()
        processor.save_cleaned_file()
        logger.info("Data cleaning completed.")
    list_of_files = list_all_files("./cleaned_tweet/train/positive")
    doc_list = [read_file(file) for file in list_of_files]
    tf_idf = TfIdfVector(doc_list)
    result = tf_idf.transform(doc_list[0])
    print(doc_list[0])
    print("TF-IDF vector shape for the first document:", result.shape)
    print("TF-IDF vector for the first document:", result)


if __name__ == "__main__":
    main()
