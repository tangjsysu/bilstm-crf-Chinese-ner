import lstm_config

train_file_path = lstm_config.train_file_path
test_file_path = lstm_config.test_file_path
def data_processor(data_dir):
    with open(data_dir, "r", encoding="utf-8") as file:
        data = file.read()
    sentences = data.split('\n\n')
    # 处理文本列表，将每个句子转换成词和标签的元组列表
    word_lists = []
    tag_lists = []
    for sentence in sentences:
        sent = sentence.split('\n')
        words = []
        tags = []
        for word in sent:
            w = word.split('\t')
            if len(w) == 2:
                words.append(w[0])
                tags.append(w[1])
        word_lists.append(words)
        tag_lists.append(tags)
    word_lists = word_lists[: -1]
    tag_lists = tag_lists[:-1]
        #word_tag = tuple([words, tags])
        #processed_data.append(word_tag)

    return(word_lists,tag_lists)

training_data,tag_list = data_processor('../data/train.txt')
testing_data, test_tag = data_processor('../data/test.txt')



word2ix = {}
tag2ix = {}
test_word2ix = {}
for sentence in training_data:
    for word in sentence:
        if word not in word2ix:
            word2ix[word] = len(word2ix)+1

for tags in tag_list:
    for tag in tags:
        if tag not in tag2ix:
            tag2ix[tag] = len(tag2ix)+1

for sentence in testing_data:
    for word in sentence:
        if word not in test_word2ix:
            test_word2ix[word] = len(test_word2ix)+1

tag2ix[lstm_config.START_TAG] = len(tag2ix)
tag2ix[lstm_config.STOP_TAG] = len(tag2ix)

word2ix['<PAD>'] = 0
tag2ix['<PAD>'] = 0
test_word2ix['<PAD>'] = 0




