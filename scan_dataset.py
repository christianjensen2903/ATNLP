import torch
from torch.utils.data import Dataset, DataLoader
from enum import Enum

SOS_token = 0
EOS_token = 1
OOV_token = 2

class ScanSplit(Enum):
    SIMPLE_SPLIT = 'simple_split'
    LENGTH_SPLIT = 'length_split'
    FEW_SHOT_SPLIT = 'few_shot_split'
    ADD_PRIM_JUMP_SPLIT = 'add_prim_split'
    ADD_PRIM_TURNLEFT_SPLIT = 'add_prim_split' # shouldn't have same value => if condition won't happen

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS", OOV_token: 'OOV'}
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
        indexes = [self.word2index.get(word,OOV_token) for word in sentence.split()]
        return indexes


    def sentence_from_indexes(self, indexes: list):
        """Get sentence from word ids"""
        return ' '.join([self.index2word[index] for index in indexes])

    def tensor_from_sentence(self, sentence:str):
        """Convert sentence to torch tensor"""
        indexes = self.indexes_from_sentence(sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


class ScanDataset(Dataset):
    def __init__(self, split: ScanSplit, input_lang: Lang, output_lang: Lang, train: bool = True):
        
        self.input_lang = input_lang
        self.output_lang = output_lang


        self.X, self.y = self._get_data(split, train)


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


    def convert_to_tensor(self, X, y):
        input_tensor = self.input_lang.tensor_from_sentence(X)
        target_tensor = self.output_lang.tensor_from_sentence(y)
        return (input_tensor, target_tensor)


    def convert_to_string(self, X, y):
        input_string = self.input_lang.sentence_from_indexes(X)
        target_string = self.output_lang.sentence_from_indexes(y)
        return (input_string, target_string)
    

    def _get_data(self, split: ScanSplit, train: bool = True):
        """Retrieve the right data for the selected split"""
        
        if split == ScanSplit.SIMPLE_SPLIT:
            X_train, y_train = self._extract_data_from_file('SCAN/simple_split/tasks_train_simple.txt')
            X_test, y_test = self._extract_data_from_file('SCAN/simple_split/tasks_test_simple.txt')
        elif split == ScanSplit.LENGTH_SPLIT:
            X_train, y_train = self._extract_data_from_file('SCAN/length_split/tasks_train_length.txt')
            X_test, y_test = self._extract_data_from_file('SCAN/length_split/tasks_test_length.txt')
        elif split == ScanSplit.ADD_PRIM_JUMP_SPLIT:
            X_train, y_train = self._extract_data_from_file('SCAN/add_prim_split/tasks_train_addprim_jump.txt')
            X_test, y_test = self._extract_data_from_file('SCAN/add_prim_split/tasks_test_addprim_jump.txt')
        elif split == ScanSplit.ADD_PRIM_TURNLEFT_SPLIT:
            X_train, y_train = self._extract_data_from_file('SCAN/add_prim_split/tasks_train_addprim_turn_left.txt')
            X_test, y_test = self._extract_data_from_file('SCAN/add_prim_split/tasks_test_addprim_turn_left.txt')
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

        return X,y
        
    def _extract_data_from_file(self, filepath: str):
        """Get X and y from SCAN file"""
        with open(filepath) as f:
            txt_data = f.readlines()

        # Format is in IN: ... OUT: ...
        lead_token = 'IN:'
        split_token = 'OUT:'

        # Split at OUT and remove IN
        txt_data = [sen.strip(lead_token).split(split_token) for sen in txt_data]

        in_txt = [sen[0] for sen in txt_data]
        out_txt = [sen[1] for sen in txt_data]

        return in_txt, out_txt