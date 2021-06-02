import time
import torch
import torch.nn as nn
from .nce.index_linear import IndexLinear

def build_unigram_noise(freq):
    """build the unigram noise from a list of frequency
    Parameters:
        freq: a tensor of #occurrences of the corresponding index
    Return:
        unigram_noise: a torch.Tensor with size ntokens,
        elements indicate the probability distribution
    """
    total = freq.sum()
    noise = freq / total
    assert abs(noise.sum() - 1) < 0.001
    return noise

class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.embedding.reset_parameters()
        self.gru_layer1 = nn.GRU(config.embed_dim, config.gru_layer1_hidden_dim // 2, batch_first=True, bidirectional=True)
        self.gru_layer2 = nn.GRU(config.gru_layer1_hidden_dim, config.gru_layer2_hidden_dim // 2, batch_first=True, bidirectional=True)

        if self.config.run_mode == 'pretrain':
            # pass
            self.attention = nn.Sequential(
                nn.Linear(config.gru_layer2_hidden_dim, config.pre_classifier_dim),
                nn.Tanh(),
                nn.Linear(config.pre_classifier_dim, 1),
                nn.Softmax(1)
            )
            # self.target_forward_prediction = nn.Linear(config.gru_layer2_hidden_dim // 2, config.vocab_size)
            # self.target_backward_prediction = nn.Linear(config.gru_layer2_hidden_dim // 2, config.vocab_size)
            self.target_sentiment = nn.Linear(config.gru_layer2_hidden_dim, config.num_sentes)
            self.polarity = nn.Linear(config.gru_layer2_hidden_dim, config.num_classes)
            self.forward_proj = nn.Linear(config.gru_layer2_hidden_dim // 2, config.pre_classifier_dim)
            self.backward_proj = nn.Linear(config.gru_layer2_hidden_dim // 2, config.pre_classifier_dim)

            # losses
            self.ce_fn = torch.nn.CrossEntropyLoss()
            self.ce_seq_fn = torch.nn.CrossEntropyLoss(ignore_index=0, reduce=False)

            # noise for soise sampling in NCE
            # print("-" * 20)
            # frequence = torch.randint(100, (config.vocab_size,)) + 1
            frequence = config.vocab_count
            # print(frequence)
            noise = build_unigram_noise(
                # torch.FloatTensor(corpus.vocab.idx2count)
                frequence
            )
            # print(noise.sum())
            # print("-"*20)
            # noise = torch.randint(100, (config.vocab_size,)).float()
            norm_term = 'auto' if config.norm_term == -1 else config.norm_term
            self.nce_fn_forward = IndexLinear(self.config.pre_classifier_dim,
                                 num_classes=self.config.vocab_size,
                                 noise=noise,
                                 noise_ratio=self.config.noise_ratio,
                                 norm_term=norm_term,
                                 loss_type='nce',
                                 reduction='none')
            self.nce_fn_backward = IndexLinear(self.config.pre_classifier_dim,
                                         num_classes=self.config.vocab_size,
                                         noise=noise,
                                         noise_ratio=self.config.noise_ratio,
                                         norm_term=norm_term,
                                         loss_type='nce',
                                         reduction='none')

    def forward(self, input_ids): # input_ids: (bs, seq), mask: (bs, seq)
        # attention_mask = mask[:, :, None] * -1e10
        hidden_states = self.embedding(input_ids) # (bs, seq, dim)
        hidden_states_1, _ = self.gru_layer1(hidden_states) # (bs, seq, dim*2), _
        hidden_states_2, _ = self.gru_layer2(hidden_states_1+hidden_states) # (bs, seq, dim*2), _

        return hidden_states, hidden_states_1, hidden_states_2

    def generate_word_representation(self, input_ids):
        hidden_states, hidden_states_1, hidden_states_2 = self(input_ids)
        return torch.cat([hidden_states, hidden_states_1, hidden_states_2], -1)

    def pretrain(self, input_ids, mask, input_token_labels, labels):
        _, _, hidden_states_2 = self(input_ids)
        attention_mask =  (1 - mask[:, :, None]) * -1e10
        att_hidden_states_2 = hidden_states_2.mul(torch.softmax(self.attention(hidden_states_2).add(attention_mask), 1)).sum(1)

        # cal logits
        target_sentiment_logits = self.target_sentiment(hidden_states_2) # (bs, seq, dim)
        polarity_logits = self.polarity(att_hidden_states_2) # (bs, dim)
        input_ids_, forward_hidden, backward_hidden, mask_ = self.get_tokens_representation_for_prediction(
            input_ids, hidden_states_2, mask
        )

        # print(self.target_forward_prediction(forward_hidden).shape)
        # print(self.target_forward_prediction(backward_hidden).shape)
        # exit()
        target_forward_logits = self.forward_proj(forward_hidden)
        target_backward_logits = self.backward_proj(backward_hidden)

        # cal losses
        loss_target_forward = self.nce_fn_forward(input_ids_, target_forward_logits)
        loss_target_forward = torch.masked_select(loss_target_forward, mask_.bool())
        loss_target_backward = self.nce_fn_backward(input_ids_, target_backward_logits)
        loss_target_backward = torch.masked_select(loss_target_backward, mask_.bool())
        loss_token_prediction = loss_target_forward.mean() + loss_target_backward.mean()

        loss_polarity = self.ce_fn(polarity_logits, labels)

        loss_target_sentiment = self.ce_seq_fn(target_sentiment_logits.view(-1, self.config.num_sentes), input_token_labels.view(-1))
        loss_target_sentiment = torch.masked_select(loss_target_sentiment, mask[:,:, None].bool()).mean()

        return loss_token_prediction, loss_target_sentiment, loss_polarity

    def get_tokens_representation_for_prediction(self, input_ids, att_hidden_states_2, mask):
        # input_ids: (bs, seq); att_hidden_states: (bs, seq, dim); mask: (bs, seq)
        max_length = input_ids.size(1)
        forward_hidden = att_hidden_states_2[:, :, :self.config.gru_layer2_hidden_dim // 2]
        backward_hidden = att_hidden_states_2[:,:, self.config.gru_layer2_hidden_dim // 2:]
        return input_ids[:, 1:max_length-1], forward_hidden[:, :max_length-2], backward_hidden[:, 2:], mask[:,1:max_length-1]


class BiLSMT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(config.s_embed_dim, config.s_lstm_hidden_dim // 2,
                            bidirectional=config.s_bidirectional, batch_first=True)
        self.calssifier = nn.Sequential(
            nn.Linear(config.s_lstm_hidden_dim, config.s_pre_classifier_dim),
            nn.Tanh(),
            nn.Linear(config.s_pre_classifier_dim, config.s_num_classes)
        )
        # losses
        self.ce_fn = torch.nn.CrossEntropyLoss()

    def forward(self, embed, mask, label):
        mask = self.generate_mask(mask.sum(1), embed.shape[1]).unsqueeze(2)  # (bs, seq, 1)
        output, _ = self.lstm(embed)
        hidden_state = (output * mask).sum(1)
        logits = self.calssifier(hidden_state)
        loss = self.ce_fn(logits, label).mean()
        return logits, loss


    def generate_mask(self, input_lengths, num_classes):
        mask = torch.nn.functional.one_hot(torch.relu(input_lengths-1), num_classes=num_classes) # (bs, seq)
        return mask






