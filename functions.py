import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from load_usps import sort_mnist

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def G1_sampler(dataset, search_range = 100, pair_num = 10):
    batches = []
    checkpoint = np.random.randint(0, len(dataset) - search_range)
    for num in range(10):
        positive_list = []
        for i in range(checkpoint, checkpoint + search_range):
            if dataset[i][1] == num:
                positive_list.append(dataset[i])
        batches.append(positive_list)           #batchesのｎ番目に[(数値ｎのテンソル), 数値ｎ]が格納されている
    pairs = []
    for i in range(pair_num):
        pair_cand = random.choice(batches)
        pair = random.choices(pair_cand, k = 2)
        pairs.append(pair)

    return(pairs)




def G2_sampler(source_dataset, target_dataset, batchsize = 2, search_range = 30, pair_num = 10):
    source_batches = []
    target_batches = []
    for num in range(10):
        s_positive_list = []
        t_positive_list = []
        for i in range(search_range):
            if source_dataset[i][1] == num:
                s_positive_list.append(source_dataset[i])
            if target_dataset[i][1] == num:
                t_positive_list.append(target_dataset[i])
        source_batches.append(s_positive_list)           #batchesのｎ番目に[(数値ｎのテンソル), 数値ｎ]が格納されている
        target_batches.append(t_positive_list)
    pairs = []
    for i in range(pair_num):
        x = np.random.randint(0, 10)
        pair_cand_s = source_batches[x]
        pair_cand_t = target_batches[x]
        s_data = random.choice(pair_cand_s)
        t_data = random.choice(pair_cand_t)
        pair = [s_data, t_data]
        pairs.append(pair)

    return(pairs)




def G3_sampler(dataset, batchsize = 2, search_range = 100, pair_num = 10):
    batches = []
    checkpoint = np.random.randint(0, len(dataset) - search_range)
    for num in range(10):
        positive_list = []
        for i in range(checkpoint, checkpoint + search_range):
            if dataset[i][1] == num:
                positive_list.append(dataset[i])
        batches.append(positive_list)           #batchesのｎ番目に[(数値ｎのテンソル), 数値ｎ]が格納されている
    pairs = []
    for i in range(pair_num):
        cand_num1, cand_num2 = random.sample(batches, k = 2)
        num1 = random.choice(cand_num1)
        num2 = random.choice(cand_num2)
        pair = [num1, num2]
        pairs.append(pair)

    return(pairs)


def G4_sampler(source_dataset, target_dataset, batchsize = 2, search_range = 30, pair_num = 10):
    source_batches = []
    target_batches = []
    for num in range(10):
        s_positive_list = []
        t_positive_list = []
        for i in range(search_range):
            if source_dataset[i][1] == num:
                s_positive_list.append(source_dataset[i])
            if target_dataset[i][1] == num:
                t_positive_list.append(target_dataset[i])
        source_batches.append(s_positive_list)           #batchesのｎ番目に[(数値ｎのテンソル), 数値ｎ]が格納されている
        target_batches.append(t_positive_list)
    pairs = []
    for i in range(pair_num):
        x, y = random.sample(np.arange(0, 10).tolist(), 2)
        pair_cand_s = source_batches[x]
        pair_cand_t = target_batches[y]
        s_data = random.choice(pair_cand_s)
        t_data = random.choice(pair_cand_t)
        pair = [s_data, t_data]
        pairs.append(pair)
        

    return(pairs)

def G_sampler(s_trainset, t_trainset, net_g,  batch_num, device = "cpu"):
    G_dataloader = []
    for epoch in range(batch_num):
        #gを固定しDCDを訓練
        net_g.eval()
        G1 = G1_sampler(s_trainset)               #1~4全部len = 10(=pair_num)
        G2 = G2_sampler(sort_mnist(s_trainset, 3), t_trainset)
        G3 = G3_sampler(s_trainset)
        G4 = G4_sampler(sort_mnist(s_trainset, 3), t_trainset)
        Gs = [G1, G2, G3, G4]
        Gs_alterd = []
        Gs_concat = []
        #各データをgにかけ、特徴量を結合
        for G in Gs:
            G_alterd = G
            G_concat = []
            for i in range(len(G)):
                batch = torch.stack([G[i][0][0], G[i][1][0]], dim = 0 ).to(device)
                tmp = net_g(batch)
                for k in range(2):
                    G_alterd[i][k] = [tmp[k], G[i][k][1]]
                G_concat.append(torch.cat([tmp[0], tmp[1]], dim = 0))       
            Gs_alterd.append(G_alterd)
            Gs_concat.append(G_concat)
        #Gs_concat[i]にはGiのデータ(結合された特徴量)がpair_num個だけ配列の形で入ってる
        #Giにラベルを付与し、シャッフルする
        G_datas = []
        
        for (label, G) in enumerate(Gs_concat):
            for (i, data) in enumerate(G):
                G_datas.append([label, data])
        random.shuffle(G_datas)
        #バッチ化する
        label_batch = []
        data_batch = []
        for label, data in G_datas:
            label_batch.append(label)
            data_batch.append(data)
        label_batch = torch.tensor(label_batch)
        data_batch = torch.stack(data_batch, dim = 0)
        G_dataloader.append([data_batch, label_batch])
    return(G_dataloader)

def G1_G2_loader(s_trainset, t_trainset, net_g): #G1, G2のペアを作り、gにかけて特徴量ペアのバッチを返す関数
    G1 = G1_sampler(s_trainset)
    G2 = G2_sampler(s_trainset, t_trainset)
    batch_data = []
    for i in range(len(G1)):                        #G1, G2のペアを展開し、横向きに格納
        batch_data.append(G1[i][0][0])
        batch_data.append(G1[i][1][0])
    for i in range(len(G2)):
        batch_data.append(G2[i][0][0])
        batch_data.append(G2[i][1][0])
    batch_data = torch.stack(batch_data)
    pred = net_g(batch_data)                       #gにかける
    G1_alt_batch = []
    G2_alt_batch = []
    batch_label = []
    batch_data = pred
    for i in range(0, len(G1)*2, 2):               #特徴量ペアの形に戻し、バッチの形にする
        pair_data = torch.cat([batch_data[i], batch_data[i+1]])
        G1_alt_batch.append(pair_data)
        batch_label.append(1)
    for i in range(0, len(G2)*2, 2):
        i += len(G1)*2
        pair_data = torch.cat([batch_data[i], batch_data[i+1]])
        G2_alt_batch.append(pair_data)
        batch_label.append(2)
    G1_G2_alt_batch = G1_alt_batch + G2_alt_batch
    G1_G2_alt_batch = torch.stack(G1_G2_alt_batch)
    return(G1_G2_alt_batch, torch.tensor(batch_label))
    
def G2_G4_loader(s_trainset, t_trainset, net_g, device = "cpu"): #G2, G4のペアを作り、gにかけて特徴量ペアのバッチを返す関数
    G2 = G2_sampler(s_trainset, t_trainset)
    G4 = G4_sampler(s_trainset, t_trainset)
    batch_data = []
    for i in range(len(G2)):                        #G2, G4のペアを展開し、横向きに格納
        batch_data.append(G2[i][0][0])
        batch_data.append(G2[i][1][0])
    for i in range(len(G4)):
        batch_data.append(G4[i][0][0])
        batch_data.append(G4[i][1][0])
    batch_data = torch.stack(batch_data).to(device)
    pred = net_g(batch_data)                       #gにかける
    G2_alt_batch = []
    G4_alt_batch = []
    batch_label = []
    batch_data = pred
    for i in range(0, len(G2)*2, 2):               #特徴量ペアの形に戻し、バッチの形にする
        pair_data = torch.cat([batch_data[i], batch_data[i+1]])
        G2_alt_batch.append(pair_data)
        batch_label.append(1)        #正解と逆のラベル
    for i in range(0, len(G4)*2, 2):
        i += len(G2)*2
        pair_data = torch.cat([batch_data[i], batch_data[i+1]])
        G4_alt_batch.append(pair_data)
        batch_label.append(3)        #正解と逆のラベル
    G2_G4_alt_batch = G2_alt_batch + G4_alt_batch
    G2_G4_alt_batch = torch.stack(G2_G4_alt_batch)
    return(G2_G4_alt_batch, torch.tensor(batch_label))