import pickle
from utils import prepare_dataset
from dataloader import make_dataloader
from models import LSTMClassifier

from tqdm import tqdm
import torch, numpy as np

def train(dloader, model, criterion, optimizer):
    model.train()
    losses, acc = [], []
    for batch in tqdm(dloader):
        y = batch["label"]
        logits = model(batch)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = torch.argmax(logits, -1)
        acc.append((preds == y).float().mean().item())

    print(
        f"Train loss: {np.array(losses).mean():4.f | Train Accuracy: {np.array(acc).mean():.4f}}"
    )

@torch.no_grad()
def test(dloader, model, criterion):
    model.eval()
    losses, acc = [], []
    for batch in dloader:
        y = batch["label"]
        logits = model(batch, criterion)
        loss = criterion(logits, y)
        losses.append(loss.item())
        preds = torch.argmax(logits, -1)
        acc.append((preds == y).float().mean().item())

    print(f"Loss: {np.array(losses).mean():.4f} | Accuracy: {np.array(acc).mean():.4f}")

def save_cp(model):
    torch.save(model.state_dict(), "checkpoints/model.pt")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset, word2index = prepare_dataset("./data")
    with open("data/word2index.pkl", "wb") as f:
        pickle.dump(word2index, f)
    train_dloader = make_dataloader(dataset["train"], word2index, 20, 128, device=True)
    val_dloader = make_dataloader(dataset["val"], word2index, 20, 500, device)
    test_dloader = make_dataloader(dataset["test"], word2index, 20, 500, device)
    model = LSTMClassifier(len(word2index), 300, 512, 2, 2)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.nn.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        print(f"===Epoch {epoch} ===")
        train(train_dloader, model, criterion, optimizer)
        print("Validating...")
        test(val_dloader, model, criterion)
        print("Testing...")
        test(test_dloader, model, criterion)


if __name__ == "__main__":
    main()
