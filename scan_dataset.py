import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from torch.nn.utils.rnn import pad_sequence
from operator import itemgetter

SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3


class ScanSplit(Enum):
    SIMPLE_SPLIT = 'simple_split'
    LENGTH_SPLIT = 'length_split'
    FEW_SHOT_SPLIT = 'few_shot_split'
    ADD_PRIM_SPLIT = 'add_prim_split'


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            SOS_token: "<SOS>",
            EOS_token: "<EOS>",
            UNK_token: '<UNK>',
            PAD_token: '<PAD>'}
        self.n_words = len(self.index2word)  # Count tokens

        self.max_length = 0

    def add_sentence(self, sentence):
        """Add sentence to vocab"""
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        """Add word to vocab"""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
            self.max_length = max(len(word), self.max_length)
        else:
            self.word2count[word] += 1

    def indexes_from_sentence(self, sentence: str):
        """Get word ids from sentence"""
        indexes = [self.word2index.get(word, UNK_token) for word in sentence.split()]
        return indexes

    def sentence_from_indexes(self, indexes: list):
        """Get sentence from word ids"""
        return ' '.join([self.index2word[index] for index in indexes])

    def tensor_from_sentence(self, sentence: str):
        """Convert sentence to torch tensor"""
        indexes = self.indexes_from_sentence(sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long)


class ScanDataset(Dataset):
    def __init__(self, split: ScanSplit, input_lang: Lang, output_lang: Lang, train: bool = True, split_variation=None):

        self.input_lang = input_lang
        self.output_lang = output_lang

        self.X, self.y = self._get_data(split, split_variation, train)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        # Convert to list if only one sample
        if len(idx) == 1:
            return [self.X[idx[0]]], [self.y[idx[0]]]

        X = list(itemgetter(*idx)(self.X))
        y = list(itemgetter(*idx)(self.y))

        return X, y

    def convert_to_tensor(self, X, y):

        for i in range(len(X)):
            X[i] = self.input_lang.tensor_from_sentence(X[i])
            y[i] = self.output_lang.tensor_from_sentence(y[i])

        # collate the batch
        input_tensor, target_tensor = self.collate(X, y)
        return (input_tensor, target_tensor)

    def convert_to_string(self, X, y):
        input_string = self.input_lang.sentence_from_indexes(X)
        target_string = self.output_lang.sentence_from_indexes(y)
        return (input_string, target_string)


    def collate(self, src_batch, tgt_batch):
        """Collate a batch of data into padded sequences."""

        src_batch = pad_sequence(src_batch, padding_value=PAD_token, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_token, batch_first=True)
        return src_batch, tgt_batch

    def _get_data(self, split: ScanSplit, split_variation=None, train: bool = True):
        """Retrieve the right data for the selected split"""

        if split == ScanSplit.SIMPLE_SPLIT:
            valid_variations = ['p1', 'p2', 'p4', 'p8', 'p16', 'p32', 'p64']
            if split_variation and split_variation in valid_variations:
                X_train, y_train = self._extract_data_from_file(
                    f'size_variations/tasks_train_simple_{split_variation}.txt', split)
                X_test, y_test = self._extract_data_from_file(
                    f'size_variations/tasks_test_simple_{split_variation}.txt', split)
            elif split_variation:
                raise Exception(f'Not a valid split variation. Valid variations are: {valid_variations}')
            else:
                X_train, y_train = self._extract_data_from_file('tasks_train_simple.txt', split)
                X_test, y_test = self._extract_data_from_file('tasks_test_simple.txt', split)
        elif split == ScanSplit.LENGTH_SPLIT:

            X_train, y_train = self._extract_data_from_file('tasks_train_length.txt', split)
            X_test, y_test = self._extract_data_from_file('tasks_test_length.txt', split)
            valid_action_seq_len = [24, 25, 26, 27, 28, 30, 32, 33, 36, 40, 48]
            valid_command_len = [4, 6, 7, 8, 9]

            if split_variation in valid_action_seq_len:
                filter_idxs = [i for i, y in enumerate(y_test) if len(y.split()) == split_variation]
                X_test = [X_test[i] for i in filter_idxs]
                y_test = [y_test[i] for i in filter_idxs]

            elif split_variation in valid_command_len:
                filter_idxs = [i for i, x in enumerate(X_test) if len(x.split()) == split_variation]
                X_test = [X_test[i] for i in filter_idxs]
                y_test = [y_test[i] for i in filter_idxs]

            elif split_variation:
                raise Exception('Split variation must be an integer')

        elif split == ScanSplit.ADD_PRIM_SPLIT:
            valid_variations = ['jump', 'turn_left']
            if split_variation and split_variation in valid_variations:
                X_train, y_train = self._extract_data_from_file(f'tasks_train_addprim_{split_variation}.txt', split)
                X_test, y_test = self._extract_data_from_file(f'tasks_test_addprim_{split_variation}.txt', split)
            else:
                raise Exception(
                    f'A valid split variation must be provided for this split. Valid variations are: {valid_variations}')
        else:
            raise Exception('Split not implemented')

        if train:
            X = X_train
            y = y_train

            # Add words to vocabs
            for sen in X:
                self.input_lang.add_sentence(sen)

            for sen in y:
                self.output_lang.add_sentence(sen)
        else:
            X = X_test
            y = y_test

        return X, y

    def _extract_data_from_file(self, filepath: str, split: ScanSplit):
        """Get X and y from SCAN file"""
        with open(f'SCAN/{split.value}/{filepath}') as f:
            txt_data = f.readlines()

        # Format is in IN: ... OUT: ...
        lead_token = 'IN:'
        split_token = 'OUT:'

        # Split at OUT and remove IN
        txt_data = [sen.strip(lead_token).split(split_token) for sen in txt_data]

        in_txt = [sen[0].strip() for sen in txt_data]
        out_txt = [sen[1].strip() for sen in txt_data]

        return in_txt, out_txt
