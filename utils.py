import os, re, unicodedata, numpy as np

def unicode_to_ascii(x):
    return "".join(
        c for c in unicodedata.normalize("NFD", x) if unicodedata.category(c) != "Mn"
    )

def normalize_text(x):
    x = unicode_to_ascii(x.lower().strip())
    x = re.sub(r"([.!?])", r"\1", x)
    x = re.sub(r"[^a-zA-Z.!?]+", r" ", x)
    return x

def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

def create_vocab(lines, min_freq):
    word2index = {"<SOS>": 0, "<EOS>":1, "<PAD>":2, "<UNK>":3}
    word2count = {}
    for sample in lines:
        text = sample.split("\t")[0]
        for w in text.split():
            if w not in word2count:
                word2count[w] = 1
            else:
                word2count[w] += 1
    
    for w,c in word2count.items():
        if c >= min_freq:
            word2index[w] = len(word2index)
    return word2index

def prepare_dataset(data_dir):
    print("Preparing dataset...")
    dataset = []
    pos_path = os.path.join(data_dir, "pos.txt")
    print("Processing pos.txt...")
    pos_lines = read_txt_file(pos_path)
    for line in pos_lines:
        line = normalize_text(line)
        sample = f"{line}\t1"
        dataset.append(sample)
    neg_path = os.path.join(data_dir, "neg.txt")
    neg_lines = read_txt_file(neg_path)
    print("Processing neg.txt...")
    for line in neg_lines:
        line = normalize_text(line)
        sample = f"{line}\t0"
        dataset.append(sample)
    
    np.random.shuffle(dataset)
    word2index = create_vocab(dataset, 10)
    print(f"Vocab size:", len(word2index))

    #create train/val/test dataset
    n_train = int(len(dataset) * 0.8)
    n_val = int(len(dataset) * 0.1)
    dataset = {
        "train": dataset[:n_train],
        "val": dataset[n_train: n_train+n_val],
        "test": dataset[-n_val:]
    }
    return dataset, word2index