import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import MeCab
import re
import numpy as np
import yaml
from datetime import datetime
from logging import getLogger, StreamHandler, INFO, DEBUG
class BiLSTM_CRF(nn.Module):
    def __init__(self,batch_size,vocab_size,tag_to_ix,embedding_dim, hidden_dim,char_vocab_size,char_embedding_dim,char_hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.batch_size = batch_size
        #charlstmpart
        self.char_embedding_dim = char_embedding_dim
        self.char_hidden_dim = char_hidden_dim
        self.char_vocab_size = char_vocab_size
        self.char_embeds = nn.Embedding(char_vocab_size,char_embedding_dim,padding_idx=0)
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim//2,num_layers=1, bidirectional=True,batch_first = True)
        self.char_hidden = 0
        #wordlstmpart
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim+char_hidden_dim, hidden_dim // 2,num_layers=1, bidirectional=True,batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.hidden = 0
        # CRF part
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
    def init_hidden(self,size):
        return (torch.randn(2, size,self.hidden_dim // 2),
                torch.randn(2, size, self.hidden_dim // 2))
    def init_char_hidden(self,length):
        return (torch.randn(2,length,self.char_hidden_dim // 2),
                torch.randn(2,length,self.char_hidden_dim // 2))
    def _get_char_lstm_features(self, charlist):
        self.char_hidden = self.init_char_hidden(charlist.size(0)*charlist.size(1))
        char_embeds = self.char_embeds(charlist)
        char_embeds = char_embeds.view(-1,char_embeds.size(2),self.char_embedding_dim)
        lstm_out, self.char_hidden = self.char_lstm(char_embeds, self.char_hidden)
        lstm_out = lstm_out[:,-1,:].view(charlist.size(0),charlist.size(1),-1)
        return lstm_out
    def _get_lstm_features(self, sentence,char_lstm_out):
        self.hidden = self.init_hidden(sentence.size(0))
        embeds = self.word_embeds(sentence)
        newembeds = torch.cat([char_lstm_out,embeds],dim=-1)
        lstm_out, self.hidden = self.lstm(newembeds, self.hidden)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    def _forward_get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence),1,-1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    def neg_log_likelihood(self, sentencelist,chars,tagslist):
        char_lstm_out =self._get_char_lstm_features(chars)
        feats = self._get_lstm_features(sentencelist,char_lstm_out)
        all_loss = 0
        count = 0
        for (feat,tag) in zip(feats,tagslist):
            forward_score = self._forward_alg(feat)
            gold_score = self._score_sentence(feat, tag)
            all_loss = all_loss+(forward_score - gold_score)
            count = count+1
        return all_loss/self.batch_size
    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = autograd.Variable(init_alphas)
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha
    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score
    def forward(self, sentence):
        lstm_feats = self._forward_get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
def load_data_and_labels(filename):
    sents, labels = [], []
    words, tags = [], []
    with open(filename) as f:
        for line in f:
            line = line.rstrip()
            if line:
                word, tag = line.split('\t')
                words.append(word)
                tags.append(tag)
            else:
                sents.append(words)
                labels.append(tags)
                words, tags = [], []
    return sents, labels
def transform_label(y):
    seq = []
    for labels in y:
        tag_ids = [vocab_label[tag] for tag in labels]
        seq.append(tag_ids)
    return seq
def get_char_sequences(x):
    chars = []
    for sent in x:
        chars.append([list(w) for w in sent])
    return chars
def transform_char(x):
    seq = []
    for sent in x:
        char_seq = []
        for w in sent:
            char_ids = [vocab_char.get(c, vocab_char[UNK]) for c in w]
            char_seq.append(char_ids)
        seq.append(char_seq)
    return seq
def transform_word(x):
    seq = []
    for sent in x:
        word_ids = [vocab_word.get(w, vocab_word[UNK]) for w in sent]
        seq.append(word_ids)
    return seq
def get_train_data(TRAIN_FILE):
    logger.info("========= START TO GET TOKEN ==========")
    sents,labels = load_data_and_labels(TRAIN_FILE)
    vocab = []
    for sent in sents:
        print(sent)
        break
"""
config
"""
config = yaml.load(open("config.yml", encoding="utf-8"))
TRAIN_FILE = (config["train_file"]["path"])
VALID_FILE = (config["valid_file"]["path"])
MODEL_FILE = config["bilstm+crf"]["model"]
epoch_num = int(config["bilstm+crf"]["epoch"])
batch_size = int(config["bilstm+crf"]["batch"])
word_embed_size = int(config["bilstm+crf"]["word_embed"])
char_embed_size = int(config["bilstm+crf"]["char_embed"])
word_hidden_size = int(config["bilstm+crf"]["word_hidden"])
char_hidden_size = int(config["bilstm+crf"]["char_hidden"])
word_lstm_layers = int(config["bilstm+crf"]["word_lstm_layers"])
char_lstm_layers = int(config["bilstm+crf"]["char_lstm_layers"])
save_model_path = config["save_model_path"]
UNKNOWN_TAG = ("<UNK>", 0)
PAD_TAG = ("<PAD>",2)
START_TAG = "<START>"
STOP_TAG = "<STOP>"
def main():
    get_train_data(TRAIN_FILE)
if  __name__ == '__main__':
    print(torch.cuda.is_available())
    main()
    