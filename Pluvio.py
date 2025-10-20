import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
device = 'cuda:0'
from ST4.sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation, util
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
import pickle
import re
import torch
import gc
from sentence_transformers import SentencesDataset  
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import torch.nn as nn 
from torch import Tensor
from typing import Dict, Iterable
from torch.autograd import Variable
from Utils import cuda
import torch.nn.functional as F
import math
import torch
from torch import nn, Tensor
import json
#import from the original sentence transformer .py file
# from SentenceTrans import SentenceTransformer

gc.collect()
torch.cuda.empty_cache()



'''
Collecting data from pickle files.
'''
with open('datasets/pickle_files/TrainingData.pkl', 'rb') as f:
    training = pickle.load(f)
with open('datasets/pickle_files/TestingDataLib.pkl', 'rb') as f:
    test_libs = pickle.load(f)
with open('datasets/pickle_files/TestingDataArch.pkl', 'rb') as f:
    test_arch = pickle.load(f)
with open('datasets/pickle_files/TestingDataLib_and_Arch.pkl', 'rb') as f:
    test_libsarch = pickle.load(f)
# with open('TestingDataArch_sameA.pkl', 'rb') as f:
#     sameA = pickle.load(f)
# with open('TestingDataArch_sameO.pkl', 'rb') as f:
#     sameO = pickle.load(f)
# with open('TestingDataArch_diffA.pkl', 'rb') as f:
#     diffA = pickle.load(f)
# with open('TestingDataArch_diffO.pkl', 'rb') as f:
#     diffO = pickle.load(f)
# with open('TestingDataLib_sameL.pkl', 'rb') as f:
#     sameL = pickle.load(f)
# with open('TestingDataLib_sameO.pkl', 'rb') as f:
#     sameO1 = pickle.load(f)
# with open('TestingDataLib_diffL.pkl', 'rb') as f:
#     diffL = pickle.load(f)
# with open('TestingDataLib_diffO.pkl', 'rb') as f:
#     diffO1 = pickle.load(f)


optimizations = ['O0', 'O1', 'O2', 'O3']
architectures = ["powerpc", "mips", "arm", "gcc32", "gcc"]
libs = ["busybox_unstripped", 'openssl', 'sqlite3', "coreutils", "curl", "magick", "puttygen"]

def pairGenerator(data):
    pairs = []
    for pair in data:
        
        for i in range(len(optimizations)):
            if optimizations[i] in pair[3]:
                opt0 = i
                break
                
        for i in range(len(optimizations)):
            if optimizations[i] in pair[4]:
                opt1 = i    
                break
                
        for j in range(len(architectures)):
            if architectures[j] in pair[3]:
                arc0 = j
                break
                
        for j in range(len(architectures)):
            if architectures[j] in pair[4]:
                arc1 = j   
                break
                
        for k in range(len(libs)):
            if libs[k] in pair[3]:
                lib0 = k
                break
                
        for k in range(len(libs)):
            if libs[k] in pair[4]:
                lib1 = k    
                break
        pairs.append(InputExample(texts=[pair[1], pair[0]], label=float(pair[2]), archs=[arc1, arc0], opts=[opt1, opt0], libs=[lib1, lib0]))
    return pairs




train_examples = pairGenerator(test_libsarch[:])



beta_1 = 1e-5
beta_2 = 5e-1
sent_embed_dim = 768
arch_opt_embed_dim = 8
batch_size = 16


class CosineSimilarityLoss(nn.Module):
    
    def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation
        
        self.embedding_arch = nn.Embedding(len(architectures), arch_opt_embed_dim).to(device)
        self.embedding_opts = nn.Embedding(len(optimizations), arch_opt_embed_dim).to(device)
        
        self.encoder = nn.Linear(sent_embed_dim+arch_opt_embed_dim*2, sent_embed_dim)
        
        # self.decode = nn.Sequential(
        #     nn.Linear(sent_embed_dim+arch_opt_embed_dim*2, sent_embed_dim))
        self.decode = nn.Sequential(
            nn.Linear(sent_embed_dim, sent_embed_dim))
        
    
    def reparametrize_n(self, mu, std, archs, opts, libs, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std
    
    
    def encode_embed(self, embed, archs=None, opts=None, libs=None, num_sample=1):
        # embd: [batch, sent_embed_dim]
        
        # print(archs, opts, libs, embed.shape[0])
        # embedding of the first arch/opt
        batch_size = embed.shape[0]
        embed_arch_0 = self.embedding_arch(torch.zeros(batch_size).type(torch.LongTensor).to(device))
        embed_opts_0 = self.embedding_opts(torch.zeros(batch_size).type(torch.LongTensor).to(device))
        
        if archs is None:
            embed_arch = embed_arch_0
        else:
            embed_arch = self.embedding_arch(archs)
        
        if opts is None:
            embed_opts = embed_opts_0
        else:
            embed_opts = self.embedding_opts(opts)
        
        encoded = self.encoder(torch.cat((embed, embed_arch, embed_opts), -1))
        
        mu = encoded
        
        '''
        HERE
        '''
        # std = F.softplus(encoded-10, beta=1.8, threshold=20)
        std = F.softplus(encoded-10, beta=1.8, threshold=20)
        

        encoding = self.reparametrize_n(mu, std, archs, opts, libs, num_sample)
        # logit = self.decode(torch.cat((encoding, embed_arch_0, embed_opts_0), -1))
        logit = self.decode(encoding)

        if num_sample == 1 : pass
        elif num_sample > 1 : logit = F.softmax(logit, dim=2).mean(0)

        
        return (mu, std), logit
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        with open(os.path.join(path, 'modules.json'), 'w') as fOut:
            json.dump(modules_config, fOut, indent=2)
    
    @classmethod
    def load(cls, path):
        model = MyModelDefinition(args)
        model.load_state_dict(torch.load('load/from/path/model.pth'))
    def __repr__(self):
        return "CosinSimilarityLoss"

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, archs, opts, libs):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        e0 = embeddings[0]
        e1 = embeddings[1]
        # print(e0.shape, e1.shape)
        
        arch0 = torch.Tensor([v[0] for v in archs]).type(torch.LongTensor).to(device)
        arch1 = torch.Tensor([v[1] for v in archs]).type(torch.LongTensor).to(device)
        opts0 = torch.Tensor([v[0] for v in opts]).type(torch.LongTensor).to(device)
        opts1 = torch.Tensor([v[1] for v in opts]).type(torch.LongTensor).to(device)
        lib0 = torch.Tensor([v[0] for v in libs]).type(torch.LongTensor).to(device)
        lib1 = torch.Tensor([v[1] for v in libs]).type(torch.LongTensor).to(device)
        
        (mu0, std0), logit0 = self.encode_embed(e0, arch0, opts0, lib0)
        (mu1, std1), logit1 = self.encode_embed(e1, arch1, opts1, lib1)
        
        
        output = self.cos_score_transformation(torch.cosine_similarity(
            logit0,
            logit1,
        ))
        
        information_loss = -0.5*(1+2*std0.log()-mu0.pow(2)-std0.pow(2)).sum(1).mean().div(math.log(2))
        information_loss += -0.5*(1+2*std1.log()-mu1.pow(2)-std1.pow(2)).sum(1).mean().div(math.log(2))
        
        cosine_loss = self.loss_fct(output, labels.view(-1))
        
        # cosine_loss += information_loss/2 * 1e-3
        cosine_loss += information_loss/2 * beta_1
        
        if 'probs' in sentence_features[0]:
            # rl_expected_reward = features['probs'] * 1 *  (-1 * cosine_loss)
            sentence_features = list(sentence_features)
            combined_probs = sentence_features[0]['probs'] * sentence_features[1]['probs']  # [batch, 1]

            reward = 1/cosine_loss

            rl_loss = combined_probs * reward * -1
            # REINFORCE Algorithm

            # print(output.shape,labels.shape, cosine_loss.shape, rl_loss.shape, )

            cosine_loss += rl_loss.sum() * beta_2
        
        return cosine_loss


    
    
class Removal(nn.Module):
    
    def __init__(self, word_embedding_model, max_len=1024):   
        super(Removal, self).__init__()
        
        # word_embedding_model.max_seq_length = 1024
        self.word_embedding_model_input_len_limit = word_embedding_model.max_seq_length
        self.word_embedding_model = word_embedding_model
        self.word_embedding_model.max_seq_length = max_len
        embedding_layer = list(word_embedding_model.modules())[3]
        
        
   
        self.embedding = embedding_layer#.to(device)
        # self.conv1d = nn.Conv1d(1024, 128, 1).to(device)
        # 768 in-channels, 1 out channels, window size of 10, padded to same as input length
        '''
        TUNE layer nodes (node = 3)
        '''
        self.conv1d = nn.Conv1d(768, 1, 3, padding='same')# in channel 768
        # self.conv1d = nn.Conv1d(8, 1, 2, padding='same')# in channel 768
        # self.linear = nn.Linear(in_features=64, out_features=1).to(device)
        self.activation = nn.Softmax(dim=1)#.to(device)
        # self.activation = nn.Softplus()#.to(device)
    
   
    def __repr__(self):
        return "Removal"
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
    def load_model(self, path):
        model = torch.load(path)
    
    def save(self, path):
        if os.path.isdir(path):
            path = os.path.join(path, 'removal.pth')
        torch.save(self.state_dict(), path)
        # torch.save(self, path)
        # config_path = os.path.join(os.path.dirname(path), 'removal.json')
        # with open(config_path, 'w') as fOut:
        #     json.dump(modules_config, fOut, indent=2)
        
    
    
    # @classmethod
    # def load(cls, path):
        # model = MyModelDefinition(args)
        # model.load_state_dict(torch.load('load/from/path/removal.pth'))
    def load(path):
        if os.path.isdir(path):
            path = os.path.join(path, 'removal.pth')
        model = torch.load(path)


    def forward(self, features: Dict[str, Tensor]):
        if not 'sentence_embedding' in features:
            # for k,v in features.items():
            #     print(k, v.shape, v.device)
            # print("========================")
            
            attention_mask = features['attention_mask'].to(device)#.to(device) #torch.Size([10, 1024])
            input_ids = features['input_ids'].to(device) #torch.Size([10, 1024])
            
            embd = self.embedding(input_ids).permute(0, 2, 1) # input_ids,shape = [10, 1024]
            # print(embed.shape) #[10, 768, 1024]
            # print("0", embd.shape)
            
            probs = self.conv1d(embd) # [10, 1, 1024]
            # print("1", probs.shape)
            probs = self.activation(probs)
            # print("2", probs.shape)
            # probs = torch.flatten(probs, start_dim=1)
            probs = probs.permute(0, 2, 1).squeeze(-1) #[10, 1024]
            # print("3", probs.shape)
            # probs = self.activation(probs).squeeze(-1) # 
            # print("4", probs.shape)
            # print(probs.shape) #[10, 1024]
            
            seq_len = probs.shape[1]
            if seq_len > self.word_embedding_model_input_len_limit:
                seq_len = self.word_embedding_model_input_len_limit
            
            top_k_probs, top_k_ind = probs.topk(seq_len, largest=True)
            # print(top_k_probs)
            # print(top_k_ind)
            
            input_ids = torch.gather(input_ids, 1, top_k_ind)
            attention_mask = torch.gather(attention_mask, 1, top_k_ind)
            
            features['attention_mask'] = attention_mask
            features['input_ids'] = input_ids
            
            
            '''
            TUNE
            '''
            #top_k_probs
            features['probs'] = torch.sum(top_k_probs, dim=1).unsqueeze(1) # [10, 1]
            # probs * 1 * (predicted - truth)
        return features


    
    


'''
BUILD MODEL
'''    

st = SentenceTransformer('all-mpnet-base-v2')
word_embedding_model = list(st.modules())[1]
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
Pluvio = SentenceTransformer(modules=[word_embedding_model, pooling_model])

Removal_Module = Removal(word_embedding_model)
Pluvio.removal_module = Removal_Module

train_dataset = SentencesDataset(train_examples, Pluvio)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
train_loss = CosineSimilarityLoss(model=Pluvio)

Pluvio.FIT(train_objectives=[(train_dataloader, train_loss)], epochs=5, warmup_steps=10, show_progress_bar=True)




'''
LOAD MODEL
'''
# def load_model(folder):
#     st = SentenceTransformer(os.path.join(folder, 'st'))
#     word_embedding_model = list(st.modules())[1]
#     pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
#     Pluvio = SentenceTransformer(modules=[word_embedding_model, pooling_model])
#     Removal_Module = torch.load(os.path.join(folder, 'removal'))
#     Pluvio.removal_module = Removal_Module
#     train_loss = torch.load(os.path.join(folder, 'loss'))
#     return Pluvio, train_loss
# Pluvio, train_loss = load_model(f'saved_models/PLUVIO')
# print("good")



def Evaluation(X_test):
    with torch.inference_mode():
        truth = [] # 0 or 1
        predictions = [] # 0-1

        pairs0 = [p[0] for p in X_test]
        pairs1 = [p[1] for p in X_test]
        truth = [p[2] for p in X_test]
        sen_embed0 = Pluvio.encode(pairs0)
        sen_embed1 = Pluvio.encode(pairs1)
        _, sen_logits0 = train_loss.encode_embed(torch.Tensor(sen_embed0).to(device))
        _, sen_logits1 = train_loss.encode_embed(torch.Tensor(sen_embed1).to(device))

        predictions = util.pairwise_cos_sim(sen_logits0, sen_logits1).detach().cpu()
        # print(predictions, truth)

        fpr, tpr, thresholds = roc_curve(truth, predictions)
        roc_auc = auc(fpr, tpr)

        # find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        pred_labels = predictions >= optimal_threshold

        res = {
            'auc': roc_auc,
            'acc': accuracy_score(truth, pred_labels),  # Maybe change to macro?
            'prc': precision_score(truth, pred_labels),
            'rcl': recall_score(truth, pred_labels),
            'pav': np.mean([p for t, p in zip(truth, predictions) if t == 1]),
            'pvr': np.var([p for t, p in zip(truth, predictions) if t == 1]),
            'nav': np.mean([p for t, p in zip(truth, predictions) if t == 0]),
            'nvr': np.var([p for t, p in zip(truth, predictions) if t == 0]),
            'f1': f1_score(truth, pred_labels),
            'opt': optimal_threshold,
        }
        # print("Testing based on the out of sample data")
        return res

    
#test based on the different and same libs. 
# to see if it is underfit 
# Evaluation(training[:160])
# Evaluation(test_arch[:160])
# Evaluation(test_libs[:160])
res = Evaluation(training[:])
print(res)

'''
SAVE MODEL
'''
def save_model(folder, plv, loss):
    removal = plv.removal_module
    torch.save(removal, os.path.join(folder, 'removal'))
    torch.save(loss, os.path.join(folder, 'loss'))
    delattr(plv, 'removal_module')
    plv.save(os.path.join(folder, 'st'))
    
if res['auc'] > 0.82:
    save_model(f'saved_models/Pluvio_new1', Pluvio, train_loss)
    print('Model Saved')
else:
    print('Pass')