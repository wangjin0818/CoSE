import torch
import io
import os
import csv
import sys
import pandas as pd

from sklearn.model_selection import train_test_split


class InputExample(object):
    def __init__(self, guid=None, text=None, label=None):
        self.guid = guid
        self.text = text
        self.label = label

class Sentiment140:
    NAME = 'Sentiment140'
    NUM_CLASSES = 5
    NUM_SENTES = 4
    MAX_LENGTH = 128

    def __init__(self, data_dir='data'):
        self.train = self._read_file(os.path.join(data_dir, 'sentiment140', 'train.csv'))
        self.dev = None
        self.test = self._read_file(os.path.join(data_dir, 'sentiment140', 'test.csv'))

    def _read_file(self, path):
        pd_reader = pd.read_csv(path, header=None, skiprows=0, encoding="latin-1", sep=',')
        documents = []
        for i in range(len(pd_reader[0])):
            # if i == 500: break
            document = list([pd_reader[0][i], pd_reader[1][i], pd_reader[2][i], pd_reader[3][i], pd_reader[4][i], pd_reader[5][i]])
            # print(document)
            documents.append(document)
        return documents

    def get_sentences(self):
        train = self._create_examples(self.train, 'train')
        dev = None
        test = self._create_examples(self.test, 'test')
        return tuple([train, dev, test])

    def _create_examples(self, documents, type):
        examples = []
        for (i, line) in enumerate(documents):
            guid = "%s-%s" % (type, i)
            examples.append(
                InputExample(guid=guid, text=line[-1], label=int(line[0])))
        return examples

    def get_sentiment_dict(self, data_dir='data', from_scratch=False):
        try:
            if from_scratch is True:
                raise Exception
            word_polarity = torch.load( os.path.join(data_dir, 'word_polarity.pt'))
        except:
            pd_reader = pd.read_csv(os.path.join(data_dir, 'subjclueslen1-HLTEMNLP05.tff'), header=None, skiprows=0,
                                    encoding="utf-8", sep=' ')
            word_polarity = {}
            int_polarity = {'both':3, 'positive': 2, 'negative':1, 'neutral':0}
            for i in range(len(pd_reader[0])):
                word, polarity = pd_reader[2][i].split('=')[-1], int_polarity[pd_reader[5][i].split('=')[-1]]
                word_polarity[word] = polarity
            torch.save(word_polarity, os.path.join(data_dir, 'word_polarity.pt'))
        return word_polarity


class MR:
    NAME = 'MR'
    NUM_CLASSES = 2
    MAX_LENGTH = 128

    def __init__(self, data_dir='data'):
        self.train, self.dev, self.test = self._split(self._read_file(os.path.join(data_dir, 'MR')))

    def _read_file(self, path):
        documents = []
        for label in ['pos', 'neg']:
            fname = os.path.join(path, 'rt-polarity.{}'.format(label))
            with io.open(fname, 'r', encoding="windows-1252") as f:
                texts = f.readlines()
            for text in texts:
                documents.append([text, 1 if label == 'pos' else 0])
        return documents

    def _split(self, data_sets):
        train, test = train_test_split(data_sets, test_size = 0.2, random_state = 42, shuffle=True)
        train, dev = train_test_split(train, test_size=0.25, random_state=42, shuffle=True)
        return train, dev, test

    def _create_examples(self, documents, type):
        examples = []
        for (i, line) in enumerate(documents):
            guid = "%s-%s" % (type, i)
            examples.append(
                InputExample(guid=guid, text=line[0], label=int(line[-1])))
        return examples

    def get_sentences(self):
        train = self._create_examples(self.train, 'train')
        dev = self._create_examples(self.dev, 'dev')
        test = self._create_examples(self.test, 'test')
        return tuple([train, dev, test])

class SST2:
    NAME = 'SST2'
    NUM_CLASSES = 2
    MAX_LENGTH = 128

    def __init__(self, data_dir='data'):
        self.train = self._read_file(os.path.join(data_dir, 'SST', 'binary', 'sentiment-train'))
        self.dev = self._read_file(os.path.join(data_dir, 'SST', 'binary', 'sentiment-dev'))
        self.test = self._read_file(os.path.join(data_dir, 'SST', 'binary', 'sentiment-test'))

    def _read_file(self, path):
        documents = []
        with io.open(path, 'r', encoding="windows-1252") as f:
            texts = f.readlines()
        for text in texts:
            print(text)
            documents.append([text[1:], text[0]])
        return documents

    def get_sentences(self):
        train = self._create_examples(self.train, 'train')
        dev = self._create_examples(self.train, 'dev')
        test = self._create_examples(self.test, 'test')
        return tuple([train, dev, test])

    def _create_examples(self, documents, type):
        examples = []
        for (i, line) in enumerate(documents):
            guid = "%s-%s" % (type, i)
            examples.append(
                InputExample(guid=guid, text=line[-1], label=int(line[0])))
        return examples


class SST5:
    NAME = 'SST5'
    NUM_CLASSES = 5
    MAX_LENGTH = 128

    def __init__(self, data_dir='data'):
        self.train = self._read_file(os.path.join(data_dir, 'SST', 'fine', 'sentiment-train'))
        self.dev = self._read_file(os.path.join(data_dir, 'SST', 'fine', 'sentiment-dev'))
        self.test = self._read_file(os.path.join(data_dir, 'SST', 'fine', 'sentiment-test'))

    def _read_file(self, path):
        documents = []
        with io.open(path, 'r', encoding="windows-1252") as f:
            texts = f.readlines()
        for text in texts:
            documents.append([text[1:], text[0]])
        return documents

    def get_sentences(self):
        train = self._create_examples(self.train, 'train')
        dev = self._create_examples(self.dev, 'dev')
        test = self._create_examples(self.test, 'test')
        return tuple([train, dev, test])

    def _create_examples(self, documents, type):
        examples = []
        for (i, line) in enumerate(documents):
            guid = "%s-%s" % (type, i)
            examples.append(
                InputExample(guid=guid, text=line[0], label=int(line[1])))
        return examples

# class SemEval13:
#     NAME = 'SemEval-2013'
#     NUM_CLASSES = 3
#     MAX_LENGTH = 128
#
#     def __init__(self, data_dir='data'):
#         self.train = self._read_file(os.path.join(data_dir, 'SST', 'binary', 'sentiment-train'))
#         self.dev = self._read_file(os.path.join(data_dir, 'SST', 'binary', 'sentiment-dev'))
#         self.test = self._read_file(os.path.join(data_dir, 'SST', 'binary', 'sentiment-test'))
#
#     def _read_file(self, path):
#         documents = []
#         with io.open(path, 'r', encoding="windows-1252") as f:
#             texts = f.readlines()
#         for text in texts:
#             documents.append([text[1:], test[0]])
#         return documents
#
#     def get_sentences(self):
#         train = self._create_examples(self.train, 'train')
#         dev = self._create_examples(self.train, 'dev')
#         test = self._create_examples(self.test, 'test')
#         return tuple([train, dev, test])
#
#     def _create_examples(self, documents, type):
#         examples = []
#         for (i, line) in enumerate(documents):
#             guid = "%s-%s" % (type, i)
#             examples.append(
#                 InputExample(guid=guid, text=line[-1], label=int(line[0])))
#         return examples


if __name__=="__main__":
    # test = Sentiment140(data_dir='../data').get_sentiment_dict(data_dir="../data")
    # print(test)
    train, dev, test = MR(data_dir='../data').get_sentences()
    print(train)
    print(dev)
    print(test)
    print(len(train))
    print(len(dev))
    print(len(test))
    # print(len(train))
    # print(len(test))

