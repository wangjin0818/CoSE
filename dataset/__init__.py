from .all_datasets import Sentiment140, MR, SST5, SST2

DATASET_PROCESSOR_MAP={
    "senti140": Sentiment140,
    "mr": MR,
    'sst2': SST2,
    'sst5': SST5
    # "yelp_13": YELP13_BERT,
    # "yelp_14": YELP14_BERT,
}
