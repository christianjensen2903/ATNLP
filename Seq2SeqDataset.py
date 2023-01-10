from torch.utils.data import Dataset
from operator import itemgetter
import torch
from torch.nn.utils.rnn import pad_sequence

class Lang:
    def __init__(self):

        self.sos_index = 0
        self.eos_index = 1
        self.unk_index = 2
        self.pad_index = 3
        
        self.word2count = {}
        self.index2word = {
            self.sos_index: "<SOS>",
            self.eos_index: "<EOS>",
            self.unk_index: '<UNK>',
            self.pad_index: '<PAD>'
        }

        # Reverse mapping
        self.word2index = dict((v, k) for k, v in self.index2word.items())

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
        indexes = [self.word2index.get(word, self.unk_index) for word in sentence.split()]
        return indexes

    def sentence_from_indexes(self, indexes: list):
        """Get sentence from word ids"""
        return ' '.join([self.index2word[index] for index in indexes])

    def tensor_from_sentence(self, sentence: str):
        """Convert sentence to torch tensor"""
        indexes = self.indexes_from_sentence(sentence)
        return torch.tensor(indexes, dtype=torch.long)

class Seq2SeqDataset(Dataset):
    def __init__(self, input_lang: Lang, output_lang: Lang, train: bool = True, **kwargs):
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.train = train

        self.X = []
        self.y = []

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        # Convert to list if only one sample
        if isinstance(idx, int):
            return [self.X[idx]], [self.y[idx]]
        elif len(idx) == 1:
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
        """Convert pytorch tensor to string"""
        input_string = self.input_lang.sentence_from_indexes(X)
        target_string = self.output_lang.sentence_from_indexes(y)
        return (input_string, target_string)


    def collate(self, src_batch, tgt_batch):
        """Collate a batch of data into padded sequences."""

        src_batch = pad_sequence(src_batch, padding_value=self.input_lang.pad_index, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.output_lang.pad_index, batch_first=True)
        return src_batch, tgt_batch