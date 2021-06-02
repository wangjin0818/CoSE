import os
import torch
from torch.utils.data import DataLoader
from dataset import DATASET_PROCESSOR_MAP
from collections import Counter

class Data(torch.utils.data.Dataset):
    sort_key = None
    def __init__(self, *data):
        assert all(len(data[0]) == len(d) for d in data)
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return tuple(d[index] for d in self.data)

def load_specific_datasets(config, bert_tokenizer):
    config.num_sentes = DATASET_PROCESSOR_MAP['senti140'].NUM_SENTES
    config.num_classes = DATASET_PROCESSOR_MAP['senti140'].NUM_CLASSES

    import spacy
    sp_tokenizer = spacy.load('en_core_web_trf')

    processor = DATASET_PROCESSOR_MAP[config.dataset]()
    config.s_num_classes = processor.NUM_CLASSES
    config.max_length = processor.MAX_LENGTH

    train_examples, dev_examples, test_examples = processor.get_sentences()

    train_input_ids, train_labels = [], []
    dev_input_ids, dev_labels = [], []
    test_input_ids, test_labels = [], []

    print("==loading train datasets")
    for step, example in enumerate(train_examples):
        token_ids = processing_over_one_example_special(example.text, bert_tokenizer, sp_tokenizer,
                                                              config)
        train_input_ids.append(token_ids)
        train_labels.append(example.label)
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(train_examples),
                                                          step / len(train_examples) * 100),
              end="")
    print("\rDone!".ljust(60))

    print("==loading dev datasets")
    for step, example in enumerate(dev_examples):
        token_ids = processing_over_one_example_special(example.text, bert_tokenizer, sp_tokenizer,
                                                              config)
        dev_input_ids.append(token_ids)
        dev_labels.append(example.label)
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dev_examples),
                                                          step / len(dev_examples) * 100),
              end="")
    print("\rDone!".ljust(60))

    print("==loading test datasets")
    for step, example in enumerate(test_examples):
        token_ids = processing_over_one_example_special(example.text, bert_tokenizer, sp_tokenizer,
                                                              config)
        test_input_ids.append(token_ids)
        test_labels.append(example.label)
        print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dev_examples),
                                                          step / len(dev_examples) * 100),
              end="")
    print("\rDone!".ljust(60))

    train_dataset = Data(train_input_ids, train_labels)
    dev_dataset = Data(dev_input_ids, dev_labels)
    test_dataset = Data(test_input_ids,  test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)
    return train_dataloader, dev_dataloader, test_dataloader

def processing_over_one_example_special(text, bert_tokenizer, sp_tokenizer, config):
    # [PAD] id is 0
    token_list = []
    tokens = sp_tokenizer.tokenizer(text)
    start_token = ['<s>']
    end_token = ['</s>']
    for tok in tokens:
        tok = tok.text
        tok = [t for t in bert_tokenizer.tokenize(tok)]
        token_list.extend(tok)

    # truncate_and_pad
    total_length = len(token_list)
    if total_length > config.max_length:
        token_list = start_token + token_list[:config.max_length] + end_token
    else:
        token_list = start_token + token_list + end_token + ['[PAD]'] * (config.max_length - total_length)

    input_id = bert_tokenizer.convert_tokens_to_ids(token_list)

    # return token_ids, token_labels
    return torch.tensor(input_id, dtype=torch.long, device=config.device)

def load_pretrained_datasets(config, bert_tokenizer, from_loacl=False):
    # trying to load in the local file for saving time
    try:
        if from_loacl:
            raise Exception
        train_data = torch.load(os.path.join('data', 'sentiment140', 'train_local.pt'))
        test_data = torch.load(os.path.join('data', 'sentiment140', 'test_local.pt'))
        data_config = torch.load(os.path.join('data', 'sentiment140', 'data_config.pt'))

        train_input_ids, train_token_labels, train_labels = train_data
        test_input_ids, test_token_labels, test_labels = test_data
        config.vocab_count, config.num_classes, config.num_sentes, config.max_length = data_config
        print("loading from local...Done.")
    except:
        import spacy
        sp_tokenizer = spacy.load('en_core_web_trf')

        processor = DATASET_PROCESSOR_MAP['senti140']()
        config.num_classes = processor.NUM_CLASSES
        config.num_sentes = processor.NUM_SENTES
        config.max_length = processor.MAX_LENGTH

        train_examples, _, test_examples = processor.get_sentences()
        senti_dic = processor.get_sentiment_dict()

        train_input_ids, train_token_labels, train_labels = [], [], []
        test_input_ids, test_token_labels, test_labels = [], [], []

        vocab_count = torch.zeros(config.vocab_size)
        print("==loading train datasets")
        for step, example in enumerate(train_examples):
            token_ids, token_labels = processing_over_one_example(example.text, bert_tokenizer, sp_tokenizer, senti_dic, config, vocab_count)
            train_input_ids.append(token_ids)
            train_token_labels.append(token_labels)
            train_labels.append(example.label)
            print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(train_examples),
                                                              step / len(train_examples) * 100),
                  end="")
        print("\rDone!".ljust(60))

        print("==loading test datasets")
        for step, example in enumerate(test_examples):
            token_ids, token_labels = processing_over_one_example(example.text, bert_tokenizer, sp_tokenizer, senti_dic,
                                                                  config, vocab_count)
            test_input_ids.append(token_ids)
            test_token_labels.append(token_labels)
            test_labels.append(example.label)
            print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(test_examples),
                                                              step / len(test_examples) * 100),
                  end="")
        print("\rDone!".ljust(60))

        vocab_count[bert_tokenizer.convert_tokens_to_ids("<s>")] = 0
        vocab_count[bert_tokenizer.convert_tokens_to_ids("[PAD]")] = 0
        vocab_count[bert_tokenizer.convert_tokens_to_ids("</s>")] = 0
        config.vocab_count = vocab_count

        train_data = train_input_ids, train_token_labels, train_labels
        test_data = test_input_ids, test_token_labels, test_labels
        data_config = config.vocab_count, config.num_classes, config.num_sentes, config.max_length

        torch.save(train_data,
                   os.path.join('data', 'sentiment140', 'train_local.pt'))
        torch.save(test_data,
                   os.path.join('data', 'sentiment140', 'test_local.pt'))
        torch.save(data_config,
                   os.path.join('data', 'sentiment140', 'data_config.pt'))

    train_dataset = Data(train_input_ids, train_token_labels, train_labels)
    test_dataset = Data(test_input_ids, test_token_labels, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)
    return train_dataloader, None, test_dataloader


def processing_over_one_example(text, bert_tokenizer, sp_tokenizer, senti_dic, config, counter):
    # [PAD] id is 0
    token_list = []
    token_label_list = []
    tokens = sp_tokenizer.tokenizer(text)
    start_token = ['<s>']
    end_token = ['</s>']
    start_token_label = [0]
    end_token_label = [0]
    for tok in tokens:
        tok = tok.text
        tok_label = [senti_dic[tok] if tok in senti_dic else 0]
        tok = [t for t in bert_tokenizer.tokenize(tok)]
        tok_label = tok_label * len(tok)
        token_list.extend(tok)
        token_label_list.extend(tok_label)

    # truncate_and_pad
    total_length = len(token_list)
    if total_length > config.max_length:
        token_list = start_token +  token_list[:config.max_length] + end_token
        token_label_list = start_token_label + token_label_list[:config.max_length] + end_token_label
    else:
        token_list = start_token +  token_list  + end_token + ['[PAD]'] * (config.max_length - total_length)
        token_label_list = start_token_label + token_label_list + [0] * (config.max_length - total_length) + end_token_label

    input_id = bert_tokenizer.convert_tokens_to_ids(token_list)
    for i in input_id:
        counter[i] += 1
    # counter.update(input_id)

    # return token_ids, token_labels
    return torch.tensor(input_id, dtype=torch.long, device=config.device), torch.tensor(token_label_list, dtype=torch.long, device=config.device)

def multi_acc(y, preds):
    preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc