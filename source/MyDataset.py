import torch
from torch.utils.data import Dataset, DataLoader
from lstm_config import *


class MyDataset(Dataset):
    # 初始化对象元素
    def __init__(self, datas, tags, word_2_index, tag_2_index):
        self.datas = datas
        self.tags = tags
        self.word_2_index = word_2_index
        self.tag_2_index = tag_2_index

    def __getitem__(self, index):
        data = self.datas[index]
        tag = self.tags[index]
        # for i in data:
        #     if i in word2index:
        #         data_index.append(word2index[i])
        #     else:
        #         data_index.append(0)
        data_index = [self.word_2_index[i] for i in data]
        tag_index = [self.tag_2_index[i] for i in tag]
        return data_index, tag_index

    def __len__(self):
        # 每句话的长度肯定和标签的长度一样
        assert len(self.datas) == len(self.tags)
        return len(self.tags)

    # 这里是对每个batch做数据处理,进行数据的拼接
    # 这batch_datas长度是batch_size中规定的那个维度
    # 里面包含了data和tag

    def pro_batch_data(self, batch_datas):
        global device
        datas = []
        tags = []
        batch_lens = []
        for data, tag in batch_datas:
            datas.append(data)
            tags.append(tag)
            batch_lens.append(len(data))

        batch_max_len = max(batch_lens)
        datas = [i + [self.word_2_index["<PAD>"]] * (batch_max_len - len(i)) for i in datas]
        tags = [i + [self.tag_2_index["<PAD>"]] * (batch_max_len - len(i)) for i in tags]
        return torch.tensor(datas, dtype=torch.int64, device=DEVICE), torch.tensor(tags, dtype=torch.long,
                                                                                   device=DEVICE)