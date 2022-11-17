# -*- coding: utf-8 -*-

# modules
import itertools
import sys
from typing import *

import numpy as np
from data.dataset import Dataset
from model_selection.split import train_test_split

sys.path.insert(0, 'src/si')


class KMer:
    """
    KMer is a feature extraction method commonly used to calculate the k-mer composition of each nucleotide (DNA) or peptide (aminoacids) sequence of a given dataset.
    """

    def __init__(self, k: int, alphabet: str):
        """
        It initializes the KMer algorithm.

        Parameters
        ----------
        k: int
            K-mers size.
        alphabet: str
            Biological sequence alphabet

        Attributes
        ----------
        k-mers: list
            List of all possible combinations of k-mers based on the k (k-mers size)
        """
        if k < 1:
            raise ValueError("k must be greater than 0")

        self.k = k

        if set(alphabet.upper()).issubset("ACGT") or set(alphabet.upper()).issubset("ACDEFGHIKLMNPQRSTVWY"):
            self.alphabet = alphabet.upper()
        else:
            raise TypeError('The sequence must be peptidic or from DNA.')

        self.k_mers = None

    def fit(self, dataset: Dataset) -> 'KMer':
        """
        It estimates k-mers's all possible combinations based on the k (k-mers size) from a given dataset. Returns self.
        """
        self.k_mers = [''.join(kmer) for kmer in itertools.product(
            self.alphabet,  repeat=self.k)]   # join: converts the resulting tuples from itertools to str
        return self

    def get_sequence_kmer_composition(self, seq: str) -> np.ndarray:
        """
        Helper function which returns the normalized relative frequency of each k-mer in a sequence. 
        """
        # Dictionary with the frequency for each possible k-mer initialized to zero
        dic_counts = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(seq) - self.k + 1):
            kmer = seq[i:i + self.k]  # sliding window to get k-mers
            dic_counts[kmer] += 1     # counts update of each k-mer

        # Normalize the counts
        return np.array([dic_counts[kmer] / len(seq) for kmer in self.k_mers])

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Calculates the k-mers composition for all sequences of a given dataset. It returns a new dataset object.

        Parameters
                ----------
                dataset: Dataset
                        Dataset object
        """
        sequences_kmer_composition = np.array([self.get_sequence_kmer_composition(
            seq) for seq in dataset.X[:, 0]])  # get the normalized counts for each sequence of the dataset

        return Dataset(X=sequences_kmer_composition, y=dataset.y, features=self.k_mers, label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits and transforms the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    print("--------Example 1--------")
    from linear_model.logistic_regression import LogisticRegression
    sys.path.insert(0, 'src')
    from si.io.csv_file import read_csv
    path = 'C:/Users/ASUS/Desktop/Bioinfo/2ano/Sistemas Inteligentes/si/datasets'
    tfbs = read_csv(path + '/tfbs.csv', sep=",", features=True, label=True)
    kmer = KMer(3, alphabet="ACtG")
    kmer_dataset = kmer.fit_transform(tfbs)

    from sklearn.preprocessing import StandardScaler
    kmer_dataset.X = StandardScaler().fit_transform(kmer_dataset.X)
    train, test = train_test_split(kmer_dataset, test_size=0.3, random_state=2)

    log_reg = LogisticRegression(use_adaptive_alpha=False)
    log_reg.fit(train)
    print(f"Predictions: {log_reg.predict(test)}")
    print(f"Score: {log_reg.score(test)}")

    print("--------Example 2--------")
    transporters = read_csv(path + '/transporters.csv',
                            sep=",", features=True, label=True)
    kmers = KMer(3, alphabet="ACDEFGHIKLMNPQRSTVWY")
    transporter_dataset = kmers.fit_transform(transporters)

    transporter_dataset.X = StandardScaler().fit_transform(transporter_dataset.X)
    train, test = train_test_split(
        transporter_dataset, test_size=0.3, random_state=2)

    log = LogisticRegression(use_adaptive_alpha=False)
    log.fit(train)
    print(f"Predictions: {log.predict(test)}")
    print(f"Score: {log.score(test)}")
