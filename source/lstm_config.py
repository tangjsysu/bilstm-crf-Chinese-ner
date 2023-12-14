import torch

train_file_path = 'D:/anaconda/envs/NLP/bilstm-crf/data/train.txt'
test_file_path = 'D:/anaconda/envs/NLP/bilstm-crf/data/test.txt'
START_TAG = "<START>"
STOP_TAG = "<STOP>"
BATCH_SIZE = 5
EMBEDDING_DIM = 50
HIDDEN_DIM = 50
EPOCH = 2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr=0.005
weight_decay=1e-4