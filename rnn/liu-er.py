import torch
import csv
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt
# 输入姓名 输出属于哪个国家
import gzip
import time
 
HIDDEN_SIZE = 100
BATCH_SIZE = 1024
N_LAYER = 2
N_EPOCHS = 100
N_CHARS = 128
USE_GPU = False
start = time.time()
 
#############################################################################    读入数据
class NameDataset(Dataset):
    def __init__(self,is_train_set=True):
        filename = 'data/liuer/names_train.csv' if is_train_set else 'data/liuer/names_test.csv'
        with open (filename,'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]  #第一列人名
        self.len = len(self.names)  #名字长度
        self.countries = [row[1] for row in rows]   #对应国家名
        self.country_list = list(sorted(set(self.countries))) #对国家名长度排序
        self.country_dict = self.getCountryDict()  #构造字典 key:国家名 value:index
        self.country_num = len(self.country_list)  #国家个数
 
    def __getitem__(self, index):     #必须重写__getitem__和__len__方法
        return self.names[index],self.country_dict[self.countries[index]]
 
    def __len__(self):
        return self.len
 
    def getCountryDict(self):
        country_dict = dict()
        for idx,country_name in enumerate(self.country_list,0):
            country_dict[country_name] = idx
        return country_dict
 
    def idx2country(self,index):
        return  self.country_list[index]
 
    def getCountriesNum(self):
        return self.country_num
 
trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = NameDataset(is_train_set=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
 
N_COUNTRY = trainset.getCountriesNum()
 
 
##############################################################################      模型
 
class RNNClassifier(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers=1,bidirectional=True): # input_size 128, hidden_size 100, output_size 一共18个国家
        super(RNNClassifier,self).__init__()
        self.hidden_size = hidden_size
        self.n_layers=n_layers
        self.n_directions = 2 if bidirectional else 1  #单向还是双向循环神经网络
        self.embedding = torch.nn.Embedding(input_size,hidden_size)
        self.gru = torch.nn.GRU(hidden_size,hidden_size,n_layers,bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size*self.n_directions,output_size) #如果是双向则维度*2
    def _init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers*self.n_directions,batch_size,self.hidden_size)
        return create_tensor(hidden)
    def forward(self,input,seq_lengths):
        #input shape Batchsize*SeqLen->SeqLen*Batchsize
        input = input.t()  #矩阵转置
        batch_size = input.size(1)
 
        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)
 
        # pack them up
        gru_input = pack_padded_sequence(embedding,seq_lengths)  ### make be sorted by descendent  打包变长序列
 
        output , hidden = self.gru(gru_input,hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1],hidden[-2]],dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output
 
 
############################################################################   数据处理
def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor
 
def name2list(name):   #返回ascll值和长度
    arr = [ord(c) for c in name]
    return arr, len(arr)
 
def make_tensors(names,countries):
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [sl[0] for sl in sequences_and_lengths]  #名字的ascll值
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths]) #单独把列表长度拿出来 （名字的长度）
    countries = countries.long()
 
    #make tensor of name,BatchSize x SeqLen   padding
    seq_tensor = torch.zeros(len(name_sequences),seq_lengths.max()).long()   #先做一个batchsize*max(seq_lengths)全0的张量
    for idx,(seq,seq_len) in enumerate(zip(name_sequences,seq_lengths),0):
        seq_tensor[idx,:seq_len]= torch.LongTensor(seq)   #把数据贴到全0的张量上去
 
    #sort by length to use pack_padded_sequence
    seq_lengths,perm_idx = seq_lengths.sort(dim=0,descending=True)   #sort返回排完序的序列和对应的index
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]
 
    return create_tensor(seq_tensor),\
        create_tensor(seq_lengths),\
        create_tensor(countries)
 
############################################################################     训练测试模块
def trainModel():
    total_loss = 0
    for i,(names,countries) in enumerate(trainloader,1):
        inputs,seq_lengths,target  = make_tensors(names,countries)
        output = classifier(inputs,seq_lengths)
        loss = criterion(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        total_loss+=loss.item()
        if i%10==0:
            print(f'[{time_since(start)}] Epoch{epoch}',end='')
            print(f'[{i*len(inputs)}/{ len(trainset)}]',end='')
            print(f'loss={total_loss/(i*len(inputs))}')
    return total_loss
 
def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model...")
    with torch.no_grad():
        for i,(names,countrise) in enumerate(testloader,1):
            inputs,seq_lengths,target = make_tensors(names,countrise)
            output = classifier(inputs,seq_lengths)
            pred = output.max(dim = 1,keepdim=True)[1]
            correct+=pred.eq(target.view_as(pred)).sum().item()
 
        percent = '%.2f'%(100*correct/total)
        print(f'Test set:Accuracy {correct}/{total} {percent}%')
 
    return correct/total
 
def time_since(start):
    """
    计算给定时间戳 `start` 与当前时间之间的时间差
    """
    return time.time() - start
 
if __name__=='__main__':
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)
 
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
 
 
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    epoch_list=[]
    for epoch in range(1,N_EPOCHS+1):
        trainModel()
        acc=testModel()
        acc_list.append(acc)
        epoch_list.append(epoch)
    plt.plot(epoch_list,acc_list)
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.grid()
    plt.show()
