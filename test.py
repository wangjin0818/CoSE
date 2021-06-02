# test

from transformers import BertTokenizer
import spacy
# import tensorflow as tf
# tf.nn.nce_loss

sp_tokenizer = spacy.load('en_core_web_trf')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# tokenizer.special_tokens_map_extended({"bos_t": "<s>","eos_t":"</s>"})

tokenizer.add_special_tokens({"bos_token": "<s>","eos_token":"</s>"})

print(tokenizer.tokenize("oh my goooood"))
print(tokenizer.tokenize("I had such a nice day. Too bad the rain comes in tomorrow at 5am "))
print([tok.text for tok in sp_tokenizer.tokenizer("I had such a nice day. Too bad the rain comes in tomorrow at 5am ")])
print([tok.text for tok in sp_tokenizer.tokenizer("oh my goooood")])

text = tokenizer.tokenize("<s> oh my goooood </s>")
print(text)
print(tokenizer.convert_tokens_to_ids(text))
print(tokenizer.special_tokens_map)
# tokenizer.add_special_tokens()
print(tokenizer.special_tokens_map_extended)


exit()

label = [1, 2, 3]

# label_ = []
for step, i in enumerate(text):
    print(i)
    if i.startswith("##"): label.insert(step, label[step-1])

print(text)
print(label)

from dataset.all_datasets import Sentiment140

test = Sentiment140().get_sentiment_dict()
print(text)