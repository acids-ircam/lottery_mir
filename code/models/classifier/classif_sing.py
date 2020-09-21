



import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import librosa
import librosa.display
import glob
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000
matplotlib.use('Agg') # for the server
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score



###############################################################################
### data import

def import_data(data_path,db_split,batch_size,sr=44100,seclen=3,augment='f',test_sing=['female5','female8','male4','male7','male10']):
    # /vocalset_10modes_44k with inside femalex/malex
    # each is a dic with 'play_tech':[['filename',signal]]
    # signal has been already split at 60db
    # can use data augmentation (pitch-shifting)
    
    # paper does not use a validation set -> validation = test
    
    tech_dic = {0:'vibrato',1:'straight',2:'breathy',3:'vocal_fry',4:'lip_trill',5:'trill',6:'trillo',7:'inhaled',8:'belt',9:'spoken'}
    
    train_sing = []
    for i in range(1,10):
        train_sing.append('female'+str(i))
    for i in range(1,12):
        train_sing.append('male'+str(i))
    
    for test_set in test_sing:
        train_sing.remove(test_set)
    
    train_data = []
    train_labels = []
    for train_set in train_sing:
        print('importing train set ',train_set)
        data_dic = np.load(data_path+train_set+'.npy',allow_pickle=True).item()
        for tech_id in tech_dic:
            tech = tech_dic[tech_id]
            data_li = data_dic[tech]
            if len(data_li)!=0:
#                print('import '+tech+' with label '+str(tech_id)+' amount #files = ',len(data_li))
                cat_sig = []
                for sig in data_li:
                    sig = sig[1]
                    sp_t = librosa.effects.split(sig,top_db=db_split,frame_length=2048,hop_length=512)
                    for sp_id in range(sp_t.shape[0]):
                        sig_tmp = sig[sp_t[sp_id,0]:sp_t[sp_id,1]]
                        cat_sig.append(sig_tmp)
                cat_sig = torch.from_numpy(np.concatenate(cat_sig).astype('float32'))
                N_seg = cat_sig.shape[0]//(seclen*sr)
                cat_sig = cat_sig[:N_seg*seclen*sr].view(N_seg,seclen*sr)
                labels = torch.zeros(N_seg,1).long()+tech_id
                train_data.append(cat_sig)
                train_labels.append(labels)
            else:
                print('passing '+tech)
    train_data = torch.cat(train_data)
    train_labels = torch.cat(train_labels)
    
    test_data = []
    test_labels = []
    for test_set in test_sing:
        print('importing test set ',test_set)
        data_dic = np.load(data_path+test_set+'.npy',allow_pickle=True).item()
        for tech_id in tech_dic:
            tech = tech_dic[tech_id]
            data_li = data_dic[tech]
            if len(data_li)!=0:
#                print('import '+tech+' with label '+str(tech_id)+' amount #files = ',len(data_li))
                cat_sig = []
                for sig in data_li:
                    sig = sig[1]
                    sp_t = librosa.effects.split(sig,top_db=db_split,frame_length=2048,hop_length=512)
                    for sp_id in range(sp_t.shape[0]):
                        sig_tmp = sig[sp_t[sp_id,0]:sp_t[sp_id,1]]
                        cat_sig.append(sig_tmp)
                cat_sig = torch.from_numpy(np.concatenate(cat_sig).astype('float32'))
                N_seg = cat_sig.shape[0]//(seclen*sr)
                cat_sig = cat_sig[:N_seg*seclen*sr].view(N_seg,seclen*sr)
                labels = torch.zeros(N_seg,1).long()+tech_id
                test_data.append(cat_sig)
                test_labels.append(labels)
            else:
                print('passing '+tech)
    test_data = torch.cat(test_data)
    test_labels = torch.cat(test_labels)
    
    if augment=='t':
        # take all training segments and pitch-shift (test set is unaltered)
        N_ref = train_data.shape[0]
        for step in [-0.5,-0.25,0.25,0.5]:
            print('augmenting with a pitch-shit of ',step)
            train_data_tmp = []
            for ref_id in range(N_ref):
                train_data_tmp.append(librosa.effects.pitch_shift(train_data[ref_id,:].numpy(), sr, n_steps=step))
            train_data_tmp = np.vstack(train_data_tmp)
            train_data = torch.cat([train_data,torch.from_numpy(train_data_tmp).float()])
        train_labels = torch.cat([train_labels,train_labels,train_labels,train_labels,train_labels])
    
    print('train_dataset,test_dataset sizes = ',train_data.shape[0],test_data.shape[0])
    
    train_refdataset = torch.utils.data.TensorDataset(train_data.contiguous(),train_labels.contiguous())
    train_loader = torch.utils.data.DataLoader(train_refdataset,batch_size=batch_size,shuffle=True,drop_last=True)
    train_refloader = torch.utils.data.DataLoader(train_refdataset,batch_size=batch_size,shuffle=False,drop_last=False)
    
    test_refdataset = torch.utils.data.TensorDataset(test_data.contiguous(),test_labels.contiguous())
    test_loader = torch.utils.data.DataLoader(test_refdataset,batch_size=batch_size,shuffle=True,drop_last=True)
    test_refloader = torch.utils.data.DataLoader(test_refdataset,batch_size=batch_size,shuffle=False,drop_last=False)
    
    return train_loader,test_loader,test_loader,train_refloader,test_refloader,test_refloader,tech_dic



###############################################################################
### evaluation functions

def loss_plot(plot_name,loss_log):
    plt.figure(figsize=(12,8))
    plt.suptitle('loss log, rows=train/test')
    plt.subplot(2,1,1)
    plt.plot(loss_log[:,0])
    plt.subplot(2,1,2)
    plt.plot(loss_log[:,1])
    plt.savefig(plot_name+'.png',format='png')
    plt.close()

def acc_plot(plot_name,epoch_log,train_acc_log,test_acc_log):
    plt.figure(figsize=(12,8))
    plt.suptitle('accuracy log, rows=train/test')
    plt.subplot(2,1,1)
    plt.plot(epoch_log,train_acc_log)
    plt.subplot(2,1,2)
    plt.plot(epoch_log,test_acc_log)
    plt.savefig(plot_name+'.png',format='png')
    plt.close()

def eval_scores(model,train_refloader,test_refloader,verbose=False):
    train_pred = []
    train_labels = []
    test_pred = []
    test_labels = []
    
    with torch.no_grad():
        
        train_loss = 0
        for _,minibatch in enumerate(train_refloader):
            x = minibatch[0].to(model.dev)
            labels  = minibatch[1].to(model.dev)
            rawpred = model.forward(x)
            loss = model.CEloss(rawpred,labels.squeeze())
            labels = labels.squeeze()
            class_pred = torch.argmax(rawpred,dim=-1)
            train_pred.append(class_pred.cpu().numpy())
            train_labels.append(labels.cpu().numpy())
            train_loss += loss.item()
        train_loss /= len(train_pred)
        # loss is averaged in the minibatch "(reduction='mean')", then divided by the number of minibatches
        
        test_loss = 0
        for _,minibatch in enumerate(test_refloader):
            x = minibatch[0].to(model.dev)
            labels  = minibatch[1].to(model.dev)
            rawpred = model.forward(x)
            loss = model.CEloss(rawpred,labels.squeeze())
            labels = labels.squeeze()
            class_pred = torch.argmax(rawpred,dim=-1)
            test_pred.append(class_pred.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            test_loss += loss.item()
        test_loss /= len(test_pred)
    
    train_pred = np.concatenate(train_pred)
    train_labels = np.concatenate(train_labels)
    test_pred = np.concatenate(test_pred)
    test_labels = np.concatenate(test_labels)
    
    target_labels = []
    target_names = []
    for tech_id in model.tech_dic:
        tech = model.tech_dic[tech_id]
        target_labels.append(tech_id)
        target_names.append(tech)
    
    if verbose is True:
        print('TRAINING SET')
        print('average training loss = ',train_loss)
        print(classification_report(train_labels, train_pred, labels=target_labels, target_names=target_names))
    train_acc = accuracy_score(train_labels, train_pred)
    train_pre = precision_score(train_labels, train_pred,average='weighted')
    train_rec = recall_score(train_labels, train_pred,average='weighted')
    print('average training accuracy, precision, recall = ',train_acc,train_pre,train_rec)
    
    if verbose is True:
        print('TEST SET')
        print('average test loss = ',test_loss)
        print(classification_report(test_labels, test_pred, labels=target_labels, target_names=target_names))
    test_acc = accuracy_score(test_labels, test_pred)
    test_pre = precision_score(test_labels, test_pred,average='weighted')
    test_rec = recall_score(test_labels, test_pred,average='weighted')
    print('average test accuracy, precision, recall = ',test_acc,test_pre,test_rec)
    
    return train_acc,test_acc,train_loss,test_loss



###############################################################################
### model for singing classification

class baseline_sing(nn.Module):
    def __init__(self,args):
        super(baseline_sing, self).__init__()
        print('singing technique classification model')
        c1=16
        c2=8
        c3=32
        h1=32
        h1_norm='t'
        dp=0.2
        
        self.batch_size = args.batch_size
        self.sr = 44100
        self.h1_norm = h1_norm
        self.dp = dp
        
        convlayers = []
        # conv1
        convlayers.append(nn.Conv1d(1,c1,128,stride=1,padding=0))
        convlayers.append(nn.ReLU())
        convlayers.append(nn.BatchNorm1d(c1))
        convlayers.append(nn.MaxPool1d(64, stride=8, padding=0))
        self.conv1 = nn.Sequential(*convlayers)
        
        convlayers = []
        # conv2
        convlayers.append(nn.Conv1d(c1,c2,64,stride=1,padding=0))
        convlayers.append(nn.ReLU())
        convlayers.append(nn.BatchNorm1d(c2))
        convlayers.append(nn.MaxPool1d(64, stride=8, padding=0))
        if dp!=0: # dropout from 2nd layer
            convlayers.append(nn.Dropout(p=dp))
        self.conv2 = nn.Sequential(*convlayers)
        
        convlayers = []
        # conv3
        convlayers.append(nn.Conv1d(c2,c3,256,stride=1,padding=0))
        convlayers.append(nn.ReLU())
        convlayers.append(nn.BatchNorm1d(c3))
        convlayers.append(nn.MaxPool1d(64, stride=8, padding=0))
        if dp!=0:
            convlayers.append(nn.Dropout(p=dp))
        self.conv3 = nn.Sequential(*convlayers)
        
        denselayers = []
        # dense1 == input is [batch,c3,217] if no padding + default settings
        denselayers.append(nn.Linear(c3*217,h1))
        denselayers.append(nn.ReLU())
        if h1_norm=='t': # else no BatchNorm1d
            denselayers.append(nn.BatchNorm1d(h1))
        if dp!=0: # dropout until last hidden layer
            denselayers.append(nn.Dropout(p=dp))
        # dense2 = ouput prediction layer
        denselayers.append(nn.Linear(h1,10))
        self.denselayers = nn.Sequential(*denselayers)
        
        self.CEloss = nn.CrossEntropyLoss(reduction='mean')
        
        self.conv3[0].unprunable = True
        self.conv3[2].unprunable = True
        self.conv3[3].unprunable = True
        self.denselayers[-1].unprunable = True
        self.CEloss.unprunable = True
    
    def cl_pred(self,x):
        # expects x of shape [batch,3*44100]
        bs = x.shape[0]
        h1 = self.conv1(x.unsqueeze(1))
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        rawpred = self.denselayers(h3.view(bs,-1).contiguous()) # [bs,10]
        return rawpred
    
    def cl_loss(self,pred,labels):
        # pred is [batch,10] ; labels is [batch,1] long in 0,9
        loss = self.CEloss(pred.contiguous(),labels.view(-1).contiguous())
        return loss
    
    def forward_w_labels(self,x,labels):
        pred = self.cl_pred(x.contiguous())
        loss = self.cl_loss(pred.contiguous(),labels.contiguous())
        return pred,loss
    
    def import_data(self):
        
        data_path = '/fast-1/datasets/vocalset_10modes_44k/'
        
        db_split = 50
        augment = 'f'
        train_loader,valid_loader,test_loader,self.train_refloader,self.valid_refloader,self.test_refloader,self.tech_dic = \
            import_data(data_path,db_split,self.batch_size,sr=44100,seclen=3,augment=augment,test_sing=['female5','female8','male4','male7','male10'])
        
        return train_loader,valid_loader,test_loader







