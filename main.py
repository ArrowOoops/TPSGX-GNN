import torch
import torch_geometric
import numpy as np
from collections import defaultdict
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
import time
from torch_geometric.data import InMemoryDataset, download_url
import random
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from goto import with_goto


def judge(alist):
	if any(alist[i+1] <= alist[i] for i in range(0,len(alist)-1)):
		return False
	else:
		return True


class Net_TPSGX(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(Net_TPSGX, self).__init__()
        self.conv1 = SAGEConv(num_node_features, 16, normalize=True)
        self.conv2 = SAGEConv(16, num_classes)
        # self.fc =

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.softmax(x, dim=1)


class Net_SAGE(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(Net_SAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, 16, normalize=True)
        self.conv2 = SAGEConv(16, num_classes)
        # self.fc =

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.softmax(x, dim=1)


def load_cora(num_nodes, filepathcites, filepathcontent):
    num_feats = 3703
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    tempy = []
    with open(filepathcontent) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()

            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]
            tempy.append(label_map[info[-1]])

    adj_lists = defaultdict(set)

    asor = []
    atar = []
    with open(filepathcites) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            asor.append(node_map[info[0]])
            atar.append(node_map[info[1]])
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    x = torch.tensor(feat_data, dtype=torch.float32)
    y = torch.tensor(tempy, dtype=torch.int64)
    edge_index = torch.tensor([asor, atar], dtype=torch.int64)

    data = Data(x=x, y=y, edge_index=edge_index)
    train = []
    test = []
    train_ind = []
    for i in range(num_nodes):
        t = random.random()
        st = t >= 0.5
        stf = not st
        train.append(st)
        test.append(stf)
        if st:
            train_ind.append(i)
    data.train_mask = torch.tensor(train, dtype=torch.bool)
    data.test_mask = torch.tensor(test, dtype=torch.bool)
    data.train_idx = torch.tensor(train_ind, dtype=torch.int64)

    # print(data)
    # print(data.y.dtype)
    return data





if __name__ == '__main__':
    cnt = 0
    lSAGE = []
    lTPSGX = []
    while cnt < 20:
        print(cnt)
        fs_SAGE = []
        fs_TPSGX = []
        lr = 0.01
        wd = 5e-4

        for i in range(10):
            cur = 331 * (i + 1)
            string = "citeseer_" + str(i + 1) + "party_" + str(cur)
            filepathcites = "NewCiteseer/" + string + "cites"
            filepathcontent = "NewCiteseer/" + string + "content"
            # string = "cora_" + str(i + 1) + "party_" + str(cur)
            # filepathcites = "NewCora/" + string + "cites"
            # filepathcontent = "NewCora/" + string + "content"
            dataset = load_cora(cur, filepathcites, filepathcontent)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = Net_SAGE(3703, 6).to(device)

            data = dataset.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            for epoch in range(200):
                optimizer.zero_grad()
                start = time.time()
                out = model(data)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

                loss.backward()
                optimizer.step()
                end = time.time()
                # if (epoch + 1) % 100 == 0:
                #     print(epoch + 1, loss, end - start)

            _, pred = model(data).max(dim=1)
            correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            acc = correct / int(data.test_mask.sum())
            data = data.to('cpu')
            pred = pred.to('cpu')
            f = f1_score(data.y[data.test_mask], pred[data.test_mask].numpy(), average="micro")
            print("SAGE ", str(i + 1), f)
            fs_SAGE.append(f)
        lSAGE.append(fs_SAGE)


        for i in range(10):
            cur = 331 * (i + 1)
            string = "citeseer_" + str(i + 1) + "party_" + str(cur)
            filepathcites = "NewCiteseer/" + string + "cites"
            filepathcontent = "NewCiteseer/" + string + "content"
            # string = "cora_" + str(i + 1) + "party_" + str(cur)
            # filepathcites = "NewCora/" + string + "cites"
            # filepathcontent = "NewCora/" + string + "content"
            dataset = load_cora(cur, filepathcites, filepathcontent)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = Net_TPSGX(3703, 6).to(device)

            data = dataset.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            for epoch in range(200):
                optimizer.zero_grad()
                start = time.time()
                out = model(data)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

                loss.backward()
                optimizer.step()
                end = time.time()
                # if (epoch + 1) % 100 == 0:
                #     print(epoch + 1, loss, end - start)

            _, pred = model(data).max(dim=1)
            correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            acc = correct / int(data.test_mask.sum())
            data = data.to('cpu')
            pred = pred.to('cpu')
            f = f1_score(data.y[data.test_mask], pred[data.test_mask].numpy(), average="micro")
            print("TPSGX ", str(i + 1), f)
            fs_TPSGX.append(f)
        lTPSGX.append(fs_TPSGX)

        cnt = cnt + 1
    S0 = []
    S1 = []
    S2= []
    S3 = []
    S4 = []
    S5 = []
    S6 = []
    S7 = []
    S8 = []
    S9 = []
    for i in lSAGE:
        S0.append(i[0])
        S1.append(i[1])
        S2.append(i[2])
        S3.append(i[3])
        S4.append(i[4])
        S5.append(i[5])
        S6.append(i[6])
        S7.append(i[7])
        S8.append(i[8])
        S9.append(i[9])
    y_SAGE = []
    y_SAGE.append(np.mean(S0))
    y_SAGE.append(np.mean(S1))
    y_SAGE.append(np.mean(S2))
    y_SAGE.append(np.mean(S3))
    y_SAGE.append(np.mean(S4))
    y_SAGE.append(np.mean(S5))
    y_SAGE.append(np.mean(S6))
    y_SAGE.append(np.mean(S7))
    y_SAGE.append(np.mean(S8))
    y_SAGE.append(np.mean(S9))

    T0 = []
    T1 = []
    T2 = []
    T3 = []
    T4 = []
    T5 = []
    T6 = []
    T7 = []
    T8 = []
    T9 = []
    for i in lTPSGX:
        T0.append(i[0])
        T1.append(i[1])
        T2.append(i[2])
        T3.append(i[3])
        T4.append(i[4])
        T5.append(i[5])
        T6.append(i[6])
        T7.append(i[7])
        T8.append(i[8])
        T9.append(i[9])
    y_TPSGX = []
    y_TPSGX.append(np.mean(T0))
    y_TPSGX.append(np.mean(T1))
    y_TPSGX.append(np.mean(T2))
    y_TPSGX.append(np.mean(T3))
    y_TPSGX.append(np.mean(T4))
    y_TPSGX.append(np.mean(T5))
    y_TPSGX.append(np.mean(T6))
    y_TPSGX.append(np.mean(T7))
    y_TPSGX.append(np.mean(T8))
    y_TPSGX.append(np.mean(T9))


    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    f = open("citeseer20test", "w")
    for i in y_SAGE:
        f.write(str(i))
        f.write('\t')
    f.write('\n')
    for i in y_TPSGX:
        f.write(str(i))
        f.write('\t')
    f.close()


    plt.plot(x, y_SAGE, c='black', linestyle='-', marker='*', linewidth=1.0)
    plt.plot(x, y_TPSGX, c='red', linestyle='-', marker='o', linewidth=1.0)
    plt.legend(['GraphSAGE', 'TPSGX'])

    plt.xlabel("Number of data holders", fontsize=12)
    plt.ylabel("F1 score", fontsize=12)
    plt.tick_params(axis='both', labelsize=10)
    plt.xticks(x)
    strname = str(cnt) + '.pdf'
    plt.savefig(strname, bbox_inches='tight')

    plt.show()






