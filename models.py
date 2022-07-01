import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, n_classes):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim, n_layers)
        self.out = torch.nn.Linear(hidden_dim, n_classes)
    
    def foward(self, x):
        embedded = self.embedding_layer(x["input_ids"])
        pcked = nn.utils.rnn.pack_padded_sequence(
            embedded,
            x["length"],
            enforce_sorted=False,
            batch_first=True,
            )
        output, _ = self.rnn(pcked)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output,
            padding_value=2,
            batch_first=True,
            )
        output = output[range(len(output)), x["length"] -1]
        return self.out(output)
    
    def foward_nopacked(self, x):
        embedded = self.embedding_layer(x["input_ids"])
        output, _ = self.rnn(embedded)
        return self.out(output[:, -1])
