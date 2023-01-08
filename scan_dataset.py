from Seq2SeqDataset import Seq2SeqDataset, Lang
from enum import Enum

class ScanSplit(Enum):
    SIMPLE_SPLIT = 'simple_split'
    LENGTH_SPLIT = 'length_split'
    FEW_SHOT_SPLIT = 'few_shot_split'
    ADD_PRIM_SPLIT = 'add_prim_split'


class ScanDataset(Seq2SeqDataset):
    def __init__(self, split: ScanSplit, input_lang: Lang, output_lang: Lang, train: bool = True, split_variation=None):
        super().__init__(input_lang, output_lang, train)

        self.X, self.y = self._get_data(split, split_variation, train)

        # Add EOS and SOS tokens
        self.X = [f'<SOS> {x} <EOS>' for x in self.X]
        self.y = [f'<SOS> {y} <EOS>' for y in self.y]



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

def get_data(split: ScanSplit, split_variation: str = None):
    """Get the SCAN dataset for the selected split"""
    input_lang = Lang()
    output_lang = Lang()

    train_dataset = ScanDataset(
        split=split,
        split_variation=split_variation,
        input_lang=input_lang,
        output_lang=output_lang,
        train=True
    )

    test_dataset = ScanDataset(
        split=split,
        split_variation=split_variation,
        input_lang=input_lang,
        output_lang=output_lang,
        train=False
    )

    return train_dataset, test_dataset