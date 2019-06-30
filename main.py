import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from load_usps import load_usps
from model import classifier, DCD, Encoder, Encoder_2
from functions import G1_sampler, G2_sampler, G3_sampler, G4_sampler, G_sampler, G2_G4_loader

#device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_ep_init_gh = 30
num_ep_init_DCD = 200
num_ep_train = 200
data_per_class = 3

batch_size = 10000          #mnist, uspsの分類タスクのときに使うバッチ数
pair_num = 10            #DCDを訓練するときのバッチサイズはpair_num x 4になる
DCD_batchnum = 40        #DCDの学習におけるバッチ数（g,hの訓練におけるDCDロスの計算のバッチ数でもある）
adv_gh_datanum = 1000    #part3のg,hのクラス分類ロスを計算する際のデータ数


data_transform = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.ToTensor()
])
s_trainset = datasets.MNIST('tmp', download = True, train = True, transform = data_transform)
s_testset = datasets.MNIST('tmp', train = False, transform = data_transform)
s_trainloader = DataLoader(s_trainset, batch_size = batch_size, shuffle = True)
s_testloader = DataLoader(s_testset, batch_size = batch_size, shuffle = True)
t_trainset, t_testset = load_usps(data_per_class) #transformの指定は禁止
t_trainloader = DataLoader(t_trainset, batch_size = batch_size, shuffle = True)
t_testloader = DataLoader(t_testset, batch_size = 64, shuffle = True)

net_g = Encoder()
net_h = classifier()
net_DCD = DCD()
loss_func = torch.nn.CrossEntropyLoss()  #損失関数は共通

#ソースにおいてgとhを訓練
print("part 1 : initial training for g and h")
optimizer = torch.optim.Adam(list(net_g.parameters())+list(net_h.parameters()), lr = 0.001)   #optimizerが両者を更新
net_g = net_g.to(device)
net_h = net_h.to(device)
net_DCD = net_DCD.to(device)
if not device == "cpu":
    net_g = nn.DataParallel(net_g)
    net_h = nn.DataParallel(net_h)
    net_DCD = nn.DataParallel(net_DCD)

for epoch in range(num_ep_init_gh):
    for data, label in s_trainloader:
        data ,label = data.to(device), label.to(device)
        optimizer.zero_grad()
        pred = net_h(net_g(data))
        loss = loss_func(pred, label)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print("epoch{} has finished".format(epoch))
torch.save(net_g.state_dict(), "model_g")
torch.save(net_h.state_dict(), "model_h")
with torch.no_grad():
    acc = 0
    total = 0
    for te_data, te_label in s_testloader:
        te_data, te_label = te_data.to(device), te_label.to(device)
        output = net_h(net_g(te_data))
        pred = torch.argmax(output, dim = 1)
        acc += (pred == te_label).sum().item() / len(te_label)
    acc = acc / len(s_testloader)
    print("accuracy in initial train of g and h(source):{}".format(acc))
    acc = 0
    total = 0
    for te_data, te_label in t_testloader:
        te_label = te_label.type(torch.LongTensor)
        te_data, te_label = te_data.to(device), te_label.to(device)
        output = net_h(net_g(te_data))
        pred = torch.argmax(output, dim = 1)
        acc += (pred == te_label).sum().item() / len(te_label)
    acc = acc / len(t_testloader)
    print("accuracy in initial train of g and h(target):{}".format(acc))

#DCDの初期training
print("part 2 : initial training for DCD")
optimizer_D = torch.optim.Adam(net_DCD.parameters(), lr = 0.01)
for epoch in range(num_ep_init_DCD):
    G_dataloader = G_sampler(s_trainset, t_trainset, net_g, DCD_batchnum)
    #G_dataloader:バッチサイズ４０(=pair_num * 4)がDCD_batchnum個だけ入ってる [[[datax40][labelx40]]xDCD_batchnum]
    loss_sum = 0
    for data, label in G_dataloader:
        data ,label = data.to(device), label.to(device)
        optimizer_D.zero_grad()
        pred = net_DCD(data)
        loss = loss_func(pred, label)
        loss.backward()
        optimizer_D.step()
        loss_sum += loss.item()

    if epoch % 10 == 0:
        print("epoch{} has finished".format(epoch))
        #テストデータ作成・検証
        with torch.no_grad():
            acc = 0
            total = 0
            G_testloader = G_sampler(s_testset, t_testset, net_g, DCD_batchnum)
            for te_data, te_label in G_testloader:
                te_data, te_label = te_data.to(device), te_label.to(device)
                output = net_DCD(te_data)
                pred = torch.argmax(output, dim = 1)
                acc += (pred == te_label).sum().item() / len(te_label)
        acc = acc / len(G_testloader)
        print("accuracy in test:{}".format(acc))
torch.save(net_DCD.state_dict(), "model_DCD")
print("accuracy in intial train of DCD:{}".format(acc))

#DCDとg,hのadversarial training
print("part 3 : adversarial training of g&h and DCD")
dcd_test_acc = []             #結果保存用の配列
cls_s_test_acc = []
cls_t_test_acc = []

for epoch in range(num_ep_train):
    #optimizer_g = torch.optim.Adam(net_g.parameters(), lr = 0.001)
    optimizer_g_h = torch.optim.Adam(list(net_g.parameters()) + list(net_h.parameters()),lr = 0.001)
    optimizer_DCD = torch.optim.Adam(net_DCD.parameters(), lr = 0.001)
    #---------DCDを固定しg,hをtrain---------------
    net_g.train().to(device)
    net_h.train().to(device)
    net_DCD.to(device)
    optimizer_g_h.zero_grad()
    #G2とG4をロード
    G2_G4_alterd, label = G2_G4_loader(s_trainset, t_trainset, net_g, device = device)
    G2_G4_alterd, label = G2_G4_alterd.to(device), label.to(device)
    #DCDに識別されるロスを計算
    dcd_pred = net_DCD(G2_G4_alterd)
    loss_dcd = loss_func(dcd_pred, label)
    #分類ロスを計算
    # ソースにおける分類ロス
    s_sample = random.choices(s_trainset, k=adv_gh_datanum)
    s_data = []
    s_label = []
    for i in range(1000):
        s_data.append(s_sample[i][0])
        s_label.append(s_sample[i][1])
    s_data = torch.stack(s_data).to(device)
    s_label = torch.LongTensor(s_label).to(device)
    s_pred = net_h(net_g(s_data))
    s_loss = loss_func(s_pred, s_label)
    #ターゲットにおける分類ロス
    t_sample = random.choices(t_trainset, k=adv_gh_datanum)
    t_data = []
    t_label = []
    for i in range(1000):
        t_data.append(t_sample[i][0])
        t_label.append(t_sample[i][1])
    t_data = torch.stack(t_data).to(device)
    t_label = torch.LongTensor(t_label).to(device)
    t_pred = net_h(net_g(t_data))
    t_loss = loss_func(t_pred, t_label)
    
    loss_sum = s_loss + t_loss + 0.2*loss_dcd
    loss_sum.backward()
    optimizer_g_h.step()
    
    #---------g,h を固定しDCDをtrain-------------
    net_g.eval()
    net_h.eval()
    net_DCD.train()
    optimizer_DCD = torch.optim.Adam(net_DCD.parameters(), lr = 0.01)
    G_dataloader = G_sampler(s_trainset, t_trainset, net_g, DCD_batchnum, device = device)
    #G_dataloader:バッチサイズ４０(=pair_num * 10)がbatch_num個だけ入ってる [[[datax40][labelx40]]xbatch_num]
    loss_s = 0
    for data, label in G_dataloader:
        data, label = data.to(device), label.to(device)
        optimizer_DCD.zero_grad()
        pred = net_DCD(data)
        loss = loss_func(pred, label)
        loss.backward()
        optimizer_DCD.step()
        loss_s += loss.item()

    if epoch % 10 == 0:
        print("epoch {} has finished".format(epoch))
        #DCDの検証
        with torch.no_grad():
            acc = 0
            total = 0
            G_testloader = G_sampler(s_testset, t_testset, net_g, DCD_batchnum, device = device)
            for te_data, te_label in G_testloader:
                te_data, te_label = te_data.to(device), te_label.to(device)
                output = net_DCD(te_data)
                pred = torch.argmax(output, dim = 1)
                acc += (pred == te_label).sum().item() / len(te_label)
        acc = acc / len(G_testloader)
        print("DCD accuracy in test:{}%".format(acc*100))
        dcd_test_acc.append(acc)

        #classifierの検証
        with torch.no_grad():
            acc = 0
            total = 0
            for te_data, te_label in s_testloader:
                te_data, te_label = te_data.to(device), te_label.to(device)
                output = net_h(net_g(te_data))
                pred = torch.argmax(output, dim = 1)
                acc += (pred == te_label).sum().item() / len(te_label)
            acc = acc / len(s_testloader)
            print("classifier test accuracy in source:{}%".format(acc*100))
            cls_s_test_acc.append(acc)

            acc = 0
            total = 0
            for te_data, te_label in t_testloader:
                te_label = te_label.type(torch.LongTensor)
                te_data, te_label = te_data.to(device), te_label.to(device)
                output = net_h(net_g(te_data))
                pred = torch.argmax(output, dim = 1)
                acc += (pred == te_label).sum().item() / len(te_label)
            acc = acc / len(t_testloader)
            print("classifier test accuracy in target:{}%".format(acc*100))
            cls_t_test_acc.append(acc)


torch.save(net_g.state_dict(), "model_g_adv")
torch.save(net_h.state_dict(), "model_h_adv")
torch.save(net_DCD.state_dict(),"model_DCD_adv")

file = open("features.txt", "w")
file.write("dcd_test_acc"+str(dcd_test_acc)+"\n")
file.write("cls_s_test_acc"+str(cls_s_test_acc)+"\n")
file.write("cls_t_test_acc"+str(cls_t_test_acc)+"\n")
file.close()

