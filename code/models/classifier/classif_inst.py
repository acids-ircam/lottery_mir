



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
from sklearn.metrics import classification_report,accuracy_score



###############################################################################
### data import with normalization and silence detection
# here we take non-silent chunks (any length), shuffle them, concatenate them

def import_data(data_path,instruments,sil_db,L,sr,Lsig,batch_size,norm_import,augment,train_ratio=0.9,verbose=True,sol_path=None,medley_path=None):
    
    train_data = []
    train_labels = []
    valid_data = []
    valid_labels = []
    test_data = []
    test_labels = []
    inst_dic = {}
    
    
    idx=0
    for instrument in instruments:
        inst_dic.update({idx:instrument})
        idx+=1
    
    print('norm_import == ',norm_import)
    
    if sol_path is None or sol_path[1]!='only':
        idx=0
        for instrument in instruments: 
            if verbose is True:
                print('training dataset is URMP - ',data_path)
                print('loading '+instrument)
            
            inst_files = glob.glob(data_path+instrument+'/*.npy')
            data_chunks = []
            orig_length = 0
            for f in inst_files:
                track = np.load(f)
                orig_length += track.shape[0]
                if norm_import is True:
                    track = track-np.mean(track)
                    track = track/(np.max(np.abs(track))*1.05)
                intervals = librosa.effects.split(track,top_db=sil_db,frame_length=L,hop_length=L)
                for i in range(intervals.shape[0]):
                    data_chunks.append(track[intervals[i,0]:intervals[i,1]])
            np.random.shuffle(data_chunks)
            data_import = np.concatenate(data_chunks)
            import_length = data_import.shape[0]
            
            train_sig = torch.from_numpy(data_import[:int(import_length*train_ratio)]).float()
            test_sig = torch.from_numpy(data_import[int(import_length*train_ratio):]).float()
            
            # reference signals == rouding to multiple of Lsig == use for evaluation scores
            train_refsig = train_sig[:int((train_sig.shape[0]//Lsig)*Lsig)].view(-1,Lsig)
            test_refsig = test_sig[:int((test_sig.shape[0]//Lsig)*Lsig)].view(-1,Lsig)
            
            N_trainref = train_refsig.shape[0]
            N_testref = test_refsig.shape[0]
            if verbose is True:
                print('subset amounts to N_trainref,N_testref == ',N_trainref,N_testref)
            
            train_lb_tmp = (torch.zeros(N_trainref).long()+idx).unsqueeze(1)
            test_lb_tmp = (torch.zeros(N_testref).long()+idx).unsqueeze(1)
            
            train_data.append(train_refsig)
            train_labels.append(train_lb_tmp)
            
            valid_data.append(test_refsig[:N_testref//2,:])
            valid_labels.append(test_lb_tmp[:N_testref//2,:])
            
            test_data.append(test_refsig[N_testref//2:,:])
            test_labels.append(test_lb_tmp[N_testref//2:,:])
            
            idx+=1
    
    
    if sol_path is not None:
        idx=0
        for instrument in instruments:
            print('adding note chunks from solv4 to data_import')
            data_chunks = []
            # goes both into train and test/eval data
            sol_files = glob.glob(sol_path[0]+instrument+'/*.npy')
            print('loading '+sol_path[0]+instrument)
            if sol_path[1] is True:
                print('no ordinario')
            if sol_path[1]=='only':
                print('only training on SOL')
            for f in sol_files:
                if sol_path[1] is False or sol_path[1]=='only': # no ommit
                    track = np.load(f)
                    if norm_import is True:
                        track = track-np.mean(track)
                        track = track/(np.max(np.abs(track))*1.05)
                    track = librosa.effects.trim(track,top_db=sil_db,frame_length=L,hop_length=L)[0]
                    data_chunks.append(track)
                if sol_path[1] is True and 'ord' not in f: # ommit ord
                    track = np.load(f)
                    if norm_import is True:
                        track = track-np.mean(track)
                        track = track/(np.max(np.abs(track))*1.05)
                    track = librosa.effects.trim(track,top_db=sil_db,frame_length=L,hop_length=L)[0]
                    data_chunks.append(track)
            np.random.shuffle(data_chunks)
            data_import = np.concatenate(data_chunks)
            import_length = data_import.shape[0]
            
            train_sig = torch.from_numpy(data_import[:int(import_length*train_ratio)]).float()
            test_sig = torch.from_numpy(data_import[int(import_length*train_ratio):]).float()
            
            # reference signals == rouding to multiple of Lsig == use for evaluation scores
            train_refsig = train_sig[:int((train_sig.shape[0]//Lsig)*Lsig)].view(-1,Lsig)
            test_refsig = test_sig[:int((test_sig.shape[0]//Lsig)*Lsig)].view(-1,Lsig)
            
            N_trainref = train_refsig.shape[0]
            N_testref = test_refsig.shape[0]
            if verbose is True:
                print('subset amounts to N_trainref,N_testref == ',N_trainref,N_testref)
            
            train_lb_tmp = (torch.zeros(N_trainref).long()+idx).unsqueeze(1)
            test_lb_tmp = (torch.zeros(N_testref).long()+idx).unsqueeze(1)
            
            train_data.append(train_refsig)
            train_labels.append(train_lb_tmp)
            
            valid_data.append(test_refsig[:N_testref//2,:])
            valid_labels.append(test_lb_tmp[:N_testref//2,:])
            
            test_data.append(test_refsig[N_testref//2:,:])
            test_labels.append(test_lb_tmp[N_testref//2:,:])
            
            idx+=1
    
    
    if medley_path is not None:
        idx=0
        for instrument in instruments:
            print('adding note chunks from medley '+medley_path[1]+' to data_import')
            data_chunks = []
            # goes both into train and test/eval data
            
            inst_path = medley_path[0]+'v1_numpy22k/'+instrument+'_22k_45db.npy'
            data_dic = np.load(inst_path,allow_pickle=True).item()
            print('loaded '+inst_path)
            for f in data_dic:
                track = data_dic[f]
                if norm_import is True:
                    track = track-np.mean(track)
                    track = track/(np.max(np.abs(track))*1.05)
                intervals = librosa.effects.split(track,top_db=sil_db,frame_length=L,hop_length=L)
                for i in range(intervals.shape[0]):
                    data_chunks.append(track[intervals[i,0]:intervals[i,1]])
            
            if medley_path[1]=='v1_phe':
                inst_path = medley_path[0]+'phenicx_np22k/'+instrument+'_22k_60db.npy'
                try:
                    data_dic = np.load(inst_path,allow_pickle=True).item()
                    print('loaded '+inst_path)
                    for f in data_dic:
                        track = data_dic[f]
                        if norm_import is True:
                            track = track-np.mean(track)
                            track = track/(np.max(np.abs(track))*1.05)
                        intervals = librosa.effects.split(track,top_db=sil_db,frame_length=L,hop_length=L)
                        for i in range(intervals.shape[0]):
                            data_chunks.append(track[intervals[i,0]:intervals[i,1]])
                except:
                    print('unavailable '+inst_path)
            
            np.random.shuffle(data_chunks)
            data_import = np.concatenate(data_chunks)
            import_length = data_import.shape[0]
            
            train_sig = torch.from_numpy(data_import[:int(import_length*train_ratio)]).float()
            test_sig = torch.from_numpy(data_import[int(import_length*train_ratio):]).float()
            
            # reference signals == rouding to multiple of Lsig == use for evaluation scores
            train_refsig = train_sig[:int((train_sig.shape[0]//Lsig)*Lsig)].view(-1,Lsig)
            test_refsig = test_sig[:int((test_sig.shape[0]//Lsig)*Lsig)].view(-1,Lsig)
            
            N_trainref = train_refsig.shape[0]
            N_testref = test_refsig.shape[0]
            if verbose is True:
                print('subset amounts to N_trainref,N_testref == ',N_trainref,N_testref)
            
            train_lb_tmp = (torch.zeros(N_trainref).long()+idx).unsqueeze(1)
            test_lb_tmp = (torch.zeros(N_testref).long()+idx).unsqueeze(1)
            
            train_data.append(train_refsig)
            train_labels.append(train_lb_tmp)
            
            valid_data.append(test_refsig[:N_testref//2,:])
            valid_labels.append(test_lb_tmp[:N_testref//2,:])
            
            test_data.append(test_refsig[N_testref//2:,:])
            test_labels.append(test_lb_tmp[N_testref//2:,:])
            
            idx+=1
    
    
    train_data = torch.cat(train_data,dim=0)
    train_labels = torch.cat(train_labels,dim=0)
    
    valid_data = torch.cat(valid_data,dim=0)
    valid_labels = torch.cat(valid_labels,dim=0)
    
    test_data = torch.cat(test_data,dim=0)
    test_labels = torch.cat(test_labels,dim=0)
    
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
    
    N_trainref = train_data.shape[0]
    N_validref = valid_data.shape[0]
    N_testref = test_data.shape[0]
    if verbose is True:
        print('dataset amounts to N_trainref,N_validref,N_testref == ',N_trainref,N_validref,N_testref)
    
    train_refdataset = torch.utils.data.TensorDataset(train_data.contiguous(),train_labels.contiguous())
    train_loader = torch.utils.data.DataLoader(train_refdataset,batch_size=batch_size,shuffle=True,drop_last=True)
    train_refloader = torch.utils.data.DataLoader(train_refdataset,batch_size=batch_size,shuffle=False,drop_last=False)
    
    valid_refdataset = torch.utils.data.TensorDataset(valid_data.contiguous(),valid_labels.contiguous())
    valid_loader = torch.utils.data.DataLoader(valid_refdataset,batch_size=batch_size,shuffle=True,drop_last=True)
    valid_refloader = torch.utils.data.DataLoader(valid_refdataset,batch_size=batch_size,shuffle=False,drop_last=False)
    
    test_refdataset = torch.utils.data.TensorDataset(test_data.contiguous(),test_labels.contiguous())
    test_loader = torch.utils.data.DataLoader(test_refdataset,batch_size=batch_size,shuffle=True,drop_last=True)
    test_refloader = torch.utils.data.DataLoader(test_refdataset,batch_size=batch_size,shuffle=False,drop_last=False)
    
    return train_loader,valid_loader,test_loader,train_refloader,valid_refloader,test_refloader,inst_dic



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

def eval_scores(model,train_refloader,test_refloader,inst_dic,device,verbose=False):
    model.eval()
    train_pred = []
    train_labels = []
    test_pred = []
    test_labels = []
    
    target_names = []
    target_labels = []
    for i in inst_dic:
        target_labels.append(i)
        target_names.append(inst_dic[i])
    
    with torch.no_grad():
        
        for _,minibatch in enumerate(train_refloader):
            x = minibatch[0].to(device)
            labels = minibatch[1].to(device)
            pred = model.cl_pred(x.contiguous()) # [batch,n_blocks,ninst]
            bs = pred.shape[0]
            n_blocks = pred.shape[1]
            labels = labels.unsqueeze(1).repeat(1,n_blocks,1).contiguous()
            pred = pred.view(bs*n_blocks,-1).contiguous()
            pred = F.softmax(pred,dim=-1)
            pred = torch.argmax(pred,dim=-1)
            labels = labels.view(bs*n_blocks)
            train_pred.append(pred.cpu().numpy())
            train_labels.append(labels.cpu().numpy())
        
        for _,minibatch in enumerate(test_refloader):
            x = minibatch[0].to(device)
            labels = minibatch[1].to(device)
            pred = model.cl_pred(x.contiguous()) # [batch,n_blocks,ninst]
            bs = pred.shape[0]
            n_blocks = pred.shape[1]
            labels = labels.unsqueeze(1).repeat(1,n_blocks,1).contiguous()
            pred = pred.view(bs*n_blocks,-1).contiguous()
            pred = F.softmax(pred,dim=-1)
            pred = torch.argmax(pred,dim=-1)
            labels = labels.view(bs*n_blocks)
            test_pred.append(pred.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
    
    train_pred = np.concatenate(train_pred)
    train_labels = np.concatenate(train_labels)
    test_pred = np.concatenate(test_pred)
    test_labels = np.concatenate(test_labels)
    
    if verbose is True:
        print('TRAINING SET')
        print(classification_report(train_labels, train_pred, labels=target_labels, target_names=target_names))
    train_acc = accuracy_score(train_labels, train_pred)
    print('global training accuracy = ',train_acc)
    if verbose is True:
        print('TEST SET')
        print(classification_report(test_labels, test_pred, labels=target_labels, target_names=target_names))
    test_acc = accuracy_score(test_labels, test_pred)
    print('global test accuracy = ',test_acc)
    return train_acc,test_acc



###############################################################################
### model for instrument classification

class baseline_inst(nn.Module):
    def __init__(self,args):
        super(baseline_inst, self).__init__()
        l=10
        L = 2**l
        Tsig = 1.5 # training signal length in second, rounded to multiple of 4*L (== block_size)
        print('instrument classification model')
        instruments = ['bn','cl','db','fl','hn','ob','sax','tba','tbn','tpt','va','vc','vn']
        ninst = len(instruments)
        c1=32
        c2=32
        c3=32
        h1=64
        h1_norm='t'
        dp=0.
        
        self.batch_size = args.batch_size
        self.sr = 22050
        self.Lsig = int((Tsig*self.sr)//(4*L) * (4*L))
        self.instruments = instruments
        self.ninst = len(instruments)
        self.l = l # if !=10 the layer dense1 with input [batch,c3,101] should be modified
        self.L = 2**l # window size
        self.block_size = 4*self.L
        self.h1_norm = h1_norm
        self.dp = dp
        
        convlayers = []
        # conv1
        convlayers.append(nn.Conv1d(1,c1,128,stride=1,padding=0))
        convlayers.append(nn.ReLU())
        convlayers.append(nn.BatchNorm1d(c1))
        convlayers.append(nn.MaxPool1d(16, stride=2, padding=0))
        self.conv1 = nn.Sequential(*convlayers)
        
        convlayers = []
        # conv2
        convlayers.append(nn.Conv1d(c1,c2,64,stride=1,padding=0))
        convlayers.append(nn.ReLU())
        convlayers.append(nn.BatchNorm1d(c2))
        convlayers.append(nn.MaxPool1d(32, stride=4, padding=0))
        if dp!=0: # dropout from 2nd layer
            convlayers.append(nn.Dropout(p=dp))
        self.conv2 = nn.Sequential(*convlayers)
        
        convlayers = []
        # conv3
        convlayers.append(nn.Conv1d(c2,c3,256,stride=1,padding=0))
        convlayers.append(nn.ReLU())
        convlayers.append(nn.BatchNorm1d(c3))
        convlayers.append(nn.MaxPool1d(16, stride=2, padding=0))
        if dp!=0:
            convlayers.append(nn.Dropout(p=dp))
        self.conv3 = nn.Sequential(*convlayers)
        
        denselayers = []
        # dense1 == input is [batch,c3,101] if no padding + modified settings
        denselayers.append(nn.Linear(c3*101,h1))
        
        denselayers.append(nn.ReLU())
        if h1_norm=='t': # else no BatchNorm1d
            denselayers.append(nn.BatchNorm1d(h1))
        if dp!=0: # dropout until last hidden layer
            denselayers.append(nn.Dropout(p=dp))
        # dense2 = ouput prediction layer
        denselayers.append(nn.Linear(h1,ninst))
        self.denselayers = nn.Sequential(*denselayers)
        
        self.CEloss = nn.CrossEntropyLoss(reduction='mean')
        
        self.conv3[0].unprunable = True
        self.conv3[2].unprunable = True
        self.conv3[3].unprunable = True
        self.denselayers[-1].unprunable = True
        self.CEloss.unprunable = True
    
    def cl_pred(self,x):
        # assumes x of shape [batch,Lsig] with Lsig multiple of 4*L = block_size ----> 1 prediction every 4*L
        bs = x.shape[0]
        n_blocks = x.shape[1]//self.block_size
        x = x[:,:n_blocks*self.block_size].view(bs*n_blocks,self.block_size).contiguous()
        h1 = self.conv1(x.unsqueeze(1))
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        rawpred = self.denselayers(h3.view(bs*n_blocks,-1).contiguous()) # [bs*n_blocks,ninst]
        return rawpred.view(bs,n_blocks,-1) # should be [batch,n_blocks,ninst] to compare with labels [batch,1] that are expanded to n_blocks length
    
    def cl_loss(self,pred,labels):
        # pred is [batch,n_blocks,ninst] ; labels is [batch,1] long in 0,ninst-1
        bs = pred.shape[0]
        n_blocks = pred.shape[1]
        labels = labels.unsqueeze(1).repeat(1,n_blocks,1).contiguous()
        loss = self.CEloss(pred.view(bs*n_blocks,-1).contiguous(),labels.view(bs*n_blocks).contiguous())
        return loss
    
    def forward_w_labels(self,x,labels):
        pred = self.cl_pred(x.contiguous())
        loss = self.cl_loss(pred.contiguous(),labels.contiguous())
        return pred,loss
    
    def import_data(self):
        
        data_path = '/fast-1/datasets/urmp_data/numpy22k/'
        sol_path = None#['/data/unagi0/adrien/urmp_data/URMP_sel_v4/',True]
        medley_path = ['/fast-1/datasets/sel_medley/','v1'] # v1_phe
        
        sil_db = 40
        norm_import = False
        augment = 'f'
        train_loader,valid_loader,test_loader,self.train_refloader,self.valid_refloader,self.test_refloader,self.inst_dic = \
            import_data(data_path,self.instruments,sil_db,self.L,self.sr,self.Lsig,self.batch_size,norm_import,augment,train_ratio=0.7,\
                        verbose=True,sol_path=sol_path,medley_path=medley_path)
        
        return train_loader,valid_loader,test_loader







