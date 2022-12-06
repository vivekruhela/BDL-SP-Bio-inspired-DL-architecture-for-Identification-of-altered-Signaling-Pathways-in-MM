# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import os
import itertools
import shap
from random import choice
from string import ascii_lowercase, digits
from dataclasses import dataclass
import pickle

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score,balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, matthews_corrcoef,make_scorer, classification_report
from sklearn.metrics import median_absolute_error, balanced_accuracy_score, precision_score, recall_score, f1_score

# %%
def set_seeds(seed_value, use_cuda):
    random.seed(seed_value)
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False
        
# set_seeds(random.randint(1,22000000), True)
set_seeds(21624865, True)

# %% [markdown]
# # Preparing Custom Datasets for Dataloader

# %%
class mm_mgus_dataloader(torch.utils.data.Dataset):

    def __init__(self,sfile,lfile,root_dir):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        if isinstance(sfile,str):
            self.name_frame = open(sfile).read().split('\n')[:-1]
            self.label_frame = open(lfile).read().split('\n')[:-1]
        else:
            self.name_frame = sfile
            self.label_frame = lfile
            
        self.root_dir = root_dir

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, idx):
        
        sname = os.path.join(self.root_dir, self.name_frame[idx])
        feat = pd.read_csv(sname,index_col=0, header=0).to_numpy()
        scaler = StandardScaler()
        feat  =scaler.fit_transform(feat)
        label = torch.tensor(int(self.label_frame[idx])).float()
        sample = (feat,label,self.name_frame[idx])

        return sample

# %%


# %% [markdown]
# # Adjacency Matrix Processing

# %%
def preprocess_adj(A):
    '''
    Pre-process adjacency matrix
    :param A: adjacency matrix
    :return:
    '''
    I = np.eye(A.shape[0])
    A_hat = A + I # add self-loops
    D_hat_diag = np.sum(A_hat, axis=1)
    D_hat_diag_inv_sqrt = np.power(D_hat_diag, -0.5)
    D_hat_diag_inv_sqrt[np.isinf(D_hat_diag_inv_sqrt)] = 0.
    D_hat_inv_sqrt = np.diag(D_hat_diag_inv_sqrt)
    return np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)

# %% [markdown]
# # GCN Model Architecture

# %%
adj_mat = pd.read_csv("adj_matrix_824_genes_string_database.csv", index_col=0, header=0)

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, acti=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim) # bias = False is also ok.
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        if acti:
            self.acti = nn.LeakyReLU(0.1)
        else:
            self.acti = None
    def forward(self, F):
        output = self.linear(F)
        if not self.acti:
            return output
        return self.acti(output)


class GCN(nn.Module):
    def __init__(self, input_dim_gcn, hidden_dim_gcn, output_dim_gcn, p, input_layer_linear, no_classes):
        super(GCN, self).__init__()
        self.gcn_layer1 = GCNLayer(input_dim_gcn, hidden_dim_gcn)
        self.gcn_layer2 = GCNLayer(hidden_dim_gcn, output_dim_gcn)
        self.fc1 = nn.Linear(input_layer_linear, no_classes)
        self.dropout = nn.Dropout(p)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, X, A = adj_mat):
        A = torch.from_numpy(preprocess_adj(A)).float().reshape(-1,A.shape[0],A.shape[1]).to(device)
        X = self.dropout(X.float())
        x = torch.matmul(A, X)
        x = self.gcn_layer1(x)
        x = self.dropout(x)
        x = torch.matmul(A, x)
        x = self.gcn_layer2(x)
        x = self.dropout(x)
        gcn_output = x
        x = x.reshape(-1,824)
        x = self.fc1(x)
        x = self.dropout(x)
        output = F.log_softmax(x, dim=1)
        return output, gcn_output
    
    def compute_l1_loss(self, w):
          return torch.abs(w).sum()

# %% [markdown]
# # Early Stopping

# %%
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, optimizer, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        # torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     }, self.path)

        self.val_loss_min = val_loss

# %% [markdown]
# # Training Module

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.,20.])).float()).to(device)

# %%
def train(model, device, train_loader, optimizer, epoch,log_interval = 10):
    tr_loss = []
    t1 = 0
    correct = 0
    correctly_predicted_training_samples = []
    correctly_predicted_training_samples_idx = []
    correctly_predicted_training_samples_name = []
    gcn_out = []
    model.train()
    for data, target, sample_names in train_loader:
        data = data.float().to(device)
        target =  target.long().to(device)
        optimizer.zero_grad()
        output,gcn_output = model(data)
        gcn_out.append(gcn_output)
        train_loss = loss(output, target.view(output.shape[0]))
        
#         Compute L1 loss component
        l1_weight = 0.00001
        l1_parameters = []
        for parameter in model.parameters():
            l1_parameters.append(parameter.view(-1))
        l1 = l1_weight * model.compute_l1_loss(torch.cat(l1_parameters))

#         Add L1 loss component
        train_loss += l1
        
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct_idx = torch.where(pred.eq(target.view_as(pred)) == True)
        correctly_predicted_training_samples_idx.append(correct_idx[0])
        correctly_predicted_training_samples.append([data[i] for i in correctly_predicted_training_samples_idx[-1]])
        correctly_predicted_training_samples_name.append([sample_names[i] for i in correctly_predicted_training_samples_idx[-1]])
        correct += pred.eq(target.view_as(pred)).sum().item()
        t1 += train_loss.item()
        tr_loss.append(train_loss)
        
        train_loss.backward()
        optimizer.step()

    t1 /= train_loader.__len__()
    acc = 100. * correct / len(train_loader.dataset)
#     print('Train Set: Average loss: {:.4f}'.format(t1))    
    return t1, acc, correctly_predicted_training_samples, correctly_predicted_training_samples_name, gcn_out

# %% [markdown]
# # Test Module

# %%
def test(model, device, test_loader, show_perf = True):
    model.eval()
    test_loss = 0
    correct = 0
    target2, pred2 = [], []
    output_pred_prob1 = torch.empty([0]).to(device)
    output_pred_prob = torch.empty([0,2]).to(device)
    correctly_predicted_test_samples = []
    correctly_predicted_test_samples_idx = []
    correctly_predicted_test_samples_name = []
    model.eval()
    with torch.no_grad():
        for data, target, sample_names in test_loader:
            data = data.to(device).float()
            target = target.long().to(device)
            output,_ = model(data)
            test_loss += loss(output, target.view(output.shape[0])).item()  # sum up batch loss
            pred1 = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            corerct_idx = torch.where(pred1.eq(target.view_as(pred1)) == True)
            correctly_predicted_test_samples_idx.append(corerct_idx[0])
            correctly_predicted_test_samples.append([data[i] for i in correctly_predicted_test_samples_idx[-1]])
            correctly_predicted_test_samples_name.append([sample_names[i] for i in correctly_predicted_test_samples_idx[-1]])
            correct += pred1.eq(target.view_as(pred1)).sum().item()
            target2.append([i.cpu().item() for i in target])
            pred2.append([j.cpu().item() for j in pred1])
            output_pred_prob1 = torch.cat((output_pred_prob1,output[:,1])) #taking pred prob of positive class only
            output_pred_prob = torch.cat((output_pred_prob,output))

    # print(f'output_pred_prob : {output_pred_prob}, pred2 : {pred2}')
    target2 = list(itertools.chain.from_iterable(target2))
    pred2 = list(itertools.chain.from_iterable(pred2))
    test_loss /= test_loader.__len__()
    acc = 100. * correct / len(test_loader.dataset)
    cm = confusion_matrix(target2, pred2)   
    cm = {'tn': cm[0, 0], 'fp': cm[0, 1],
          'fn': cm[1, 0], 'tp': cm[1, 1]}
    class_report = classification_report(target2,pred2,output_dict=True)
    f1_score_cal1 = class_report['0']['f1-score']
    prec1 = class_report['0']['precision']
    rec1 = class_report['0']['recall']
    f1_score_cal2 = class_report['1']['f1-score']
    prec2 = class_report['1']['precision']
    rec2 = class_report['1']['recall']
    f1_score_cal = f1_score(target2, pred2)
    f1_score_weighted = f1_score(target2, pred2,average='weighted')
    f1_score_weighted1 = f1_score(target2, pred2,average='weighted',labels=[0])
    f1_score_weighted2 = f1_score(target2, pred2,average='weighted',labels=[1])
    f1_score_micro = f1_score(target2, pred2,average='micro')
    f1_score_macro = f1_score(target2, pred2,average='macro')
    acc = accuracy_score(target2, pred2)
    acc_balanced = balanced_accuracy_score(target2, pred2)
    prec = precision_score(target2, pred2, average = 'weighted')
    rec = recall_score(target2, pred2, average = 'weighted')
    roc = roc_auc_score(target2, output_pred_prob.cpu()[:, 1], average='weighted')
    mcc = matthews_corrcoef(target2, pred2)

    perf_mat = {'accuracy':acc,
                'balanced_accuracy':acc_balanced,
                'f1_score':f1_score_cal,
                'f1_score_MM':f1_score_cal1,
                'f1_score_MGUS':f1_score_cal2,
                'f1_score_weighted':f1_score_weighted,
                'f1_score_weighted_MM':f1_score_weighted1,
                'f1_score_weighted_MGUS':f1_score_weighted2,
                'f1_score_micro':f1_score_micro,
                'f1_score_macro':f1_score_macro,
                'precision':prec,
                'precision_MM':prec1,
                'precision_MGUS':prec2,
                'recall': rec,
                'recall_MM': rec1,
                'recall_MGUS': rec2,
                'roc':roc,
                'mcc': mcc,
                'confusioin_matrix':cm
               }
    if show_perf:
        print(perf_mat)

#     print('Test set: Average loss: {:.4f}, MGUS Recall: {}/{} ({:.2f}%)'.format(
#         test_loss, cm['tp'], cm['tp']+cm['fn'],
#         100. * cm['tp'] / (cm['tp']+cm['fn'])))
    
    return test_loss, acc, perf_mat

# %%
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():        
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
             

# %%
root_dir = '/home/vivek/top_model_results/NN/all_feat_mat'
sfile = '/home/vivek/top_model_results/NN/samples_without_PB.txt'
lfile = '/home/vivek/top_model_results/NN/labels_without_PB.txt'
X = open(sfile).read().split('\n')
Y = open(lfile).read().split('\n')
# Shuffling the data
X1,Y1 =[], []
new_idx = list(np.arange(len(X)))
new_idx = random.sample(new_idx,len(new_idx))
for i in new_idx:
    X1.append(X[i])
    Y1.append(Y[i])

def save_folds(xtrain, xtest, ytrain, ytest, foldno):
    with open('/home/vivek/top_model_results/NN/test/xtrain_fold'+str(foldno)+'.txt','w') as handle:
        for i in xtrain:
            line = i + '\n'
            handle.write(line)
    with open('/home/vivek/top_model_results/NN/test/xtest_fold'+str(foldno)+'.txt','w') as handle:
        for i in xtest:
            line = i + '\n'
            handle.write(line)
    with open('/home/vivek/top_model_results/NN/test/ytrain_fold'+str(foldno)+'.txt','w') as handle:
        for i in ytrain:
            line = i + '\n'
            handle.write(line)
    with open('/home/vivek/top_model_results/NN/test/ytest_fold'+str(foldno)+'.txt','w') as handle:
        for i in ytest:
            line = i + '\n'
            handle.write(line)
# %%
epochs = 350
fold_no = 1
batch_size = 24
patience = 50
overall_tp,overall_fp,overall_tn,overall_fn = [],[],[],[]
train_loss, test_loss, test_acc, score, final_cm = {},{},{}, {}, {}
acc,balanced_acc,f1_sc,f1_score_weighted, f1_score_micro,f1_score_macro = [],[],[],[],[],[]
prec,rec,sav,roc,mcc = [],[],[],[],[]
prec_mgus,prec_mm,f1_mgus,f1_mm = [],[],[],[]
f1_wt_mgus,f1_wt_mm = [],[]
rec_mgus,rec_mm = [],[]
acc_train, acc_balanced_train = [], []
# output_pred_prob2 = []
train_loss, test_loss = {}, {}
corr_pred_train_sample_dict, corr_pred_train_sample_name_dict = {}, {}
corr_pred_test_sample_dict, corr_pred_test_sample_name_dict = {}, {}
kf = StratifiedKFold(n_splits=5, random_state = 108, shuffle=True)

for train_index, test_index in kf.split(X1, Y1):
    print(f'**********************Fold-{fold_no} Training*************************')
    model = GCN(28,7,1,0.75,824,2).to(device)
    model.apply(reset_weights)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001,
                                 weight_decay=0) #weight_decay=1e-5
    early_stopping = EarlyStopping(optimizer,
                                   patience=patience,
                                   path='GCN_model_'+str(fold_no)+'_checkpoint.pt',
                                   trace_func=warnings.warn)
    train_loss['fold_'+str(fold_no)] = []
    test_loss['fold_'+str(fold_no)] = []
    X_train = [X1[i] for i in train_index]
    y_train = [Y1[i] for i in train_index]
    X_test = [X1[i] for i in test_index]
    y_test = [Y1[i] for i in test_index]
    save_folds(X_train, X_test, y_train, y_test, fold_no)
    tr_mm_mgus_dataset = mm_mgus_dataloader(sfile = X_train, lfile = y_train, root_dir = root_dir)
    val_mm_mgus_dataset = mm_mgus_dataloader(sfile = X_test, lfile = y_test, root_dir = root_dir)
    gcn_out_per_iter_dict = {}
    for iterations in   (range(0, epochs)):
        train_loader = DataLoader(tr_mm_mgus_dataset,batch_size=batch_size,shuffle=False, num_workers=20)
        val_loader = DataLoader(val_mm_mgus_dataset,batch_size=batch_size,shuffle=False, num_workers=20)
        tr, tr_acc, corr_pred_train_sample, corr_pred_train_sample_names, gcn_out = train(model, device, train_loader, optimizer, iterations)
        if iterations == epochs-1:
            te, te_acc,perf_mat = test(model, device, val_loader, show_perf=False)
        else:
            te, te_acc,perf_mat = test(model, device, val_loader, show_perf=False)

        train_loss['fold_'+str(fold_no)].append(tr)
        test_loss['fold_'+str(fold_no)].append(te)
        gcn_out_per_iter_dict['iter_'+str(iterations)] = gcn_out


        early_stopping(te, model)

        if early_stopping.early_stop:
            print(perf_mat)
            print("############# Early stopping ###############")
            break

    overall_tp.append(perf_mat['confusioin_matrix']['tp'])
    overall_tn.append(perf_mat['confusioin_matrix']['tn'])
    overall_fp.append(perf_mat['confusioin_matrix']['fp'])
    overall_fn.append(perf_mat['confusioin_matrix']['fn'])
    acc.append(perf_mat['accuracy'])
    balanced_acc.append(perf_mat['balanced_accuracy'])
    del model
    fold_no += 1

    
final_cm['tp'] = sum(overall_tp)
final_cm['fp'] = sum(overall_fp)
final_cm['tn'] = sum(overall_tn)
final_cm['fn'] = sum(overall_fn)
print('************************************')
print('The final confusion matrix is : ',final_cm)

# %%
# Model Performance analysis
print('The mean balanced accuracy is         :',np.mean(balanced_acc))
print('The mean ROC is                       :',np.mean(roc))

# %%
plt.figure(figsize=(20,6))

a = pd.DataFrame(train_loss['fold_1'])
plt.plot(a,label='Fold-1 Train Loss')

b = pd.DataFrame(train_loss['fold_2'])
plt.plot(b,label='Fold-2 Train Loss')

c = pd.DataFrame(train_loss['fold_3'])
plt.plot(c,label='Fold-3 Train Loss')

d = pd.DataFrame(train_loss['fold_4'])
plt.plot(d,label='Fold-4Train Loss')

e = pd.DataFrame(train_loss['fold_4'])
plt.plot(e,label='FOld-5 Train Loss')
plt.xticks(np.arange(0,350,10))
# plt.yticks(np.arange(0,30,2))
plt.legend()
plt.grid()
plt.title('Training Loss Plot')
plt.savefig('Training_Loss_Curve1.png', dpi = 400)

# # %%
plt.figure(figsize=(20,6))

a = pd.DataFrame(test_loss['fold_1'])
plt.plot(a,label='Fold-1 Test Loss')

b = pd.DataFrame(test_loss['fold_2'])
plt.plot(b,label='Fold-2 Test Loss')

c = pd.DataFrame(test_loss['fold_3'])
plt.plot(c,label='Fold-3 Test Loss')

d = pd.DataFrame(test_loss['fold_4'])
plt.plot(d,label='Fold-4 Test Loss')

e = pd.DataFrame(test_loss['fold_5'])
plt.plot(e,label='Fold-5 Test Loss')
plt.xticks(np.arange(0,350,10))
# plt.yticks(np.arange(0,30,2))
plt.legend()
plt.grid()
plt.title('Validation Loss Plot')
plt.savefig('TValidation_Loss_Curve1.png', dpi = 400)

# %%
torch.cuda.empty_cache()
