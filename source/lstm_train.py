import torch
from torch import nn, optim
from lstm_config import *
from helper_function import log_sum_exp, argmax, prepare_sequence
from bilstm_crf_model import BiLSTM_CRF
from lstmprocessor import *
from MyDataset import *
from lstmprocessor import *
from tqdm import trange
from tqdm import tqdm
from sklearn.metrics import f1_score

training_data, train_tag = data_processor('../data/train.txt')
testing_data, test_tag = data_processor('../data/test.txt')


train_dataset = MyDataset(training_data, train_tag, word2ix, tag2ix)
# print("train_dataset'getItem=", train_dataset.getListItem())
train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False, collate_fn=train_dataset.pro_batch_data)

# 验证数据
dev_dataset = MyDataset(testing_data, test_tag, test_word2ix, tag2ix)
dev_dataloader = DataLoader(dev_dataset, BATCH_SIZE, shuffle=False, collate_fn=dev_dataset.pro_batch_data)

model = BiLSTM_CRF(len(word2ix), tag2ix, EMBEDDING_DIM, HIDDEN_DIM)
model = model.to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

for epoch in trange(EPOCH):
    model.train()
    for sentences, tags in tqdm(train_dataloader):
        sentences = sentences.to(DEVICE)
        tags = tags.to(DEVICE)
        sentences = sentences.reshape(-1)
        tags = tags.reshape(-1)
        # print("sentences=", sentences)
        # print("tags= ", tags)
        # 第一步，pytorch梯度累积，需要清零梯度
        model.zero_grad()
        # 第二步，得到loss
        loss = model.neg_log_likelihood(sentences, tags).cuda()
        print("crf loss=", loss)
        # 第三步，计算loss，梯度，通过optimier更新参数
        loss.backward()
        optimizer.step()

    # 开始测试
    model.eval()
    # 用于计算f1_score
    all_pre = []
    all_tag = []
    for dev_sentences, dev_tags in tqdm(dev_dataloader):
        # 预测的结果
        dev_sentences.view(-1)
        dev_tags.view(-1)
        dev_pre_score, dev_pre_tag = model.forward(dev_sentences)
        # print("dev_pre_score", dev_pre_score)
        # print("dev_pre_tag", dev_pre_tag)
        # 预测的值放入集合中用于计算f1_score,f1_score的输入是两个list和一个average函数
        all_pre.extend(dev_pre_tag)
        print("dev_pre_tags=", dev_pre_tag)
        print("dev_tags=", dev_tags)
        # 把测试集的tag拍平
        dev_tags_flat = dev_tags.detach().cpu().reshape(-1).tolist()
        # 标签值放入集合
        all_tag.extend(dev_tags_flat)
        # print("dev_data_tag", tagFromDev)
    # 计算f1_score
    score = f1_score(all_tag, all_pre, average="micro")
    print("f1_score:", score)
