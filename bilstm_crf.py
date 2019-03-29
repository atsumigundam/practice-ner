import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import re
import numpy as np
import yaml
from datetime import datetime
from logging import getLogger, StreamHandler, INFO, DEBUG
from collections import Counter
import random
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BiLSTM_CRF(nn.Module):
    def __init__(self,batch_size,word_vocab_size,tag_to_ix,word_embedding_dim, word_hidden_dim,char_vocab_size,char_embedding_dim,char_hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.batch_size = batch_size
        #charlstmpart
        self.char_embedding_dim = char_embedding_dim
        self.char_hidden_dim = char_hidden_dim
        self.char_vocab_size = char_vocab_size
        self.char_embeds = nn.Embedding(char_vocab_size,char_embedding_dim,padding_idx=PAD_TAG[1])
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim//2,num_layers=char_lstm_layers, bidirectional=True,batch_first = True)
        self.char_hidden = 0
        #wordlstmpart
        self.word_embedding_dim = word_embedding_dim
        self.word_hidden_dim = word_hidden_dim
        self.word_vocab_size = word_vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding(word_vocab_size,word_embedding_dim,padding_idx=PAD_TAG[1])
        self.word_lstm = nn.LSTM(word_embedding_dim+char_hidden_dim, word_hidden_dim // 2,num_layers=word_lstm_layers, bidirectional=True,batch_first=True)
        self.hidden2tag = nn.Linear(word_hidden_dim, self.tagset_size)
        self.hidden = 0
        # CRF part
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG[0]], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG[0]]] = -10000
    def init_word_hidden(self,size):
        return (torch.randn(2, size,self.word_hidden_dim // 2,device=device),
                torch.randn(2, size, self.word_hidden_dim // 2,device=device))
    def init_char_hidden(self,size):
        return (torch.randn(2,size,self.char_hidden_dim // 2,device=device),
                torch.randn(2,size,self.char_hidden_dim // 2,device=device))
    def _get_char_lstm_features(self, charlist):
        self.char_hidden = self.init_char_hidden(charlist.size(0)*charlist.size(1))
        char_embeds = self.char_embeds(charlist)
        char_embeds = char_embeds.view(-1,char_embeds.size(2),self.char_embedding_dim)
        lstm_out, self.char_hidden = self.char_lstm(char_embeds, self.char_hidden)
        lstm_out = lstm_out[:,-1,:].view(charlist.size(0),charlist.size(1),-1)
        return lstm_out
    def _get_lstm_features(self, sentence,char_lstm_out):
        self.word_hidden = self.init_word_hidden(sentence.size(0))
        embeds = self.word_embeds(sentence)
        newembeds = torch.cat([char_lstm_out,embeds],dim=-1)
        lstm_out, self.word_hidden = self.word_lstm(newembeds, self.word_hidden)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    def _predict_get_lstm_features(self, sentence):
        self.word_hidden = self.init_word_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence),1,-1)
        lstm_out, self.word_hidden = self.word_lstm(embeds, self.word_hidden)
        lstm_out = lstm_out.view(len(sentence), self.word_hidden_dim)
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
        init_alphas[0][self.tag_to_ix[START_TAG[0]]] = 0.
        forward_var = autograd.Variable(init_alphas)
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG[0]]]
        alpha = log_sum_exp(terminal_var)
        return alpha
    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG[0]]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG[0]], tags[-1]]
        return score
    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG[0]]] = 0
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG[0]]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG[0]]
        best_path.reverse()
        return path_score, best_path
    def forward(self, sentence):
        lstm_feats = self._predict_get_lstm_features(sentence)
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
def transform_label(y,label_to_id):
    tag_ids = [label_to_id[tag] for tag in y]
    return tag_ids
def get_char_sequences(x):
    chars = []
    for sent in x:
        chars.append([list(w) for w in sent])
    return chars
def transform_char(x,char_to_id):
    char_ids = []
    for chars in x:
        char_ids.append([char_to_id.get(c, char_to_id[UNKNOWN_TAG[0]]) for c in chars])
    return char_ids
def transform_word(x,word_to_id):
    word_ids = [word_to_id.get(w, word_to_id[UNKNOWN_TAG[0]]) for w in x]
    return word_ids
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
def get_train_data(TRAIN_FILE):
    logger.info("========= START TO GET TOKEN ==========")
    sents,labels = load_data_and_labels(TRAIN_FILE)
    word_to_id = {UNKNOWN_TAG[0]: UNKNOWN_TAG[1], PAD_TAG[0]: PAD_TAG[1]}
    char_to_id = {UNKNOWN_TAG[0]: UNKNOWN_TAG[1], PAD_TAG[0]: PAD_TAG[1]}
    label_to_id = {START_TAG[0]:START_TAG[1],PAD_TAG[0]: PAD_TAG[1],STOP_TAG[0]:STOP_TAG[1]}
    words = get_char_sequences(sents)
    for sent in sents:
        for w in sent:
            for c in w:
                if c in char_to_id:
                    continue
                char_to_id[c] = len(char_to_id)
            if w in word_to_id:
                continue
            word_to_id[w] = len(word_to_id)
    for label in labels:
        for tag in label:
            if tag in label_to_id:
                continue
            label_to_id[tag] = len(label_to_id)
    logger.debug(word_to_id)
    logger.debug(char_to_id)
    logger.debug(label_to_id)
    id_to_word = {i: v for v, i in word_to_id.items()}
    id_to_char = {i: v for v, i in char_to_id.items()}
    id_to_label = {i: v for v, i in label_to_id.items()}
    logger.debug(id_to_word)
    logger.debug(id_to_char)
    logger.debug(id_to_label)
    train_data = [[transform_word(sent,word_to_id),transform_char(word,char_to_id),transform_label(label,label_to_id)] for (sent,word,label) in zip(sents,words,labels)]
    return word_to_id,char_to_id,label_to_id,id_to_word,id_to_char,id_to_label,train_data
def make_minibatch(training_data):
    n = len(training_data)
    mini_batch_size = int(n/batch_size)
    random.shuffle(training_data)
    batch_training_data = []
    for i in range(0,n,mini_batch_size):
        if i+batch_size>n:
            batch_training_data.append(training_data[i:])
        else:
            batch_training_data.append(training_data[i:i+batch_size])
    return batch_training_data

def train(train_data,word_to_id,char_to_id,label_to_id,MODEL_FILE):
    logger.info("========= WORD_SIZE={} ==========".format(len(word_to_id)))
    logger.info("========= CHAR_SIZE={} ==========".format(len(char_to_id)))
    logger.info("========= TRAIN_SIZE={} =========".format(len(train_data)))
    logger.info("========= START_TRAIN ==========")
    model = BiLSTM_CRF(batch_size,len(word_to_id),label_to_id,word_embed_size,word_hidden_size,len(char_to_id),char_embed_size,char_hidden_size).to(device)
    model_optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    all_EPOCH_LOSS = []
    for epoch in range(epoch_num):
        total_loss = 0
        logger.info("=============== EPOCH {} {} ===============".format(epoch + 1, datetime.now()))
        batch_training_data = make_minibatch(train_data)
        for count,batch_data in enumerate(batch_training_data):
            logger.debug("===== {} / {} =====".format(count, len(batch_training_data)))
            word_data = [data[0] for data in batch_data]
            char_data = [data[1] for data in batch_data]
            label_data = [data[2] for data in batch_data]
            #padding part
            word_length= [len(seq) for seq in word_data]
            char_length =  [[len(char) for char in word] for word in char_data]
            input_word_data = [sentence if len(sentence)==max(word_length) else sentence+[word_to_id[PAD_TAG[0]] for i in range(max(word_length) - len(sentence))] for sentence in word_data]
            input_char_data = []
            max_char_length = max([max(chars) for chars in char_length])
            input_label_data = [labels if len(labels)==max(word_length) else labels+[label_to_id[PAD_TAG[0]] for i in range(max(word_length) - len(labels))] for labels in label_data]
            for (word,length) in zip(char_data,char_length):
                if len(word)==max(word_length):
                   word=[char if len(char)==max_char_length else char+[char_to_id[PAD_TAG[0]] for i in range(max_char_length - len(char))] for char in word]
                   input_char_data.append(word)                
                else:
                    word.extend([[PAD_TAG[1]]]*(max(word_length)-len(word)))
                    word=[char if len(char)==max_char_length else char+[char_to_id[PAD_TAG[0]] for i in range(max_char_length - len(char))] for char in word]
                    input_char_data.append(word)
            #train part
            input_word_data=torch.tensor(input_word_data,dtype=torch.long,device=device)
            input_char_data=torch.tensor(input_char_data,dtype=torch.long,device=device)
            input_label_data = torch.tensor(input_label_data,dtype=torch.long,device=device)
            loss = model.neg_log_likelihood(input_word_data,input_char_data,input_label_data)
            loss.backward()
            model_optimizer.step()
            total_loss += loss
            logger.info("=============== loss: %s ===============" % loss)
        total_loss = total_loss/len(batch_training_data)
        logger.info("=============== total_loss: %s ===============" % total_loss)
        all_EPOCH_LOSS.append(total_loss)
    [logger.info("================ batchnumber: {}---loss: {}=======================".format(batchnumber,loss)) for batchnumber,loss in enumerate(all_EPOCH_LOSS)]
    torch.save(model.state_dict(), MODEL_FILE)
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
PAD_TAG = ("<PAD>",1)
START_TAG = ("<START>",0)
STOP_TAG = ("<STOP>",2)
def main():
    word_to_id,char_to_id,label_to_id,id_to_word,id_to_char,id_to_label,train_data=get_train_data(TRAIN_FILE)
    train(train_data,word_to_id,char_to_id,label_to_id,MODEL_FILE)
if  __name__ == '__main__':
    print(torch.cuda.is_available())
    main()
    