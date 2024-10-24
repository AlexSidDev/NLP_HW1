import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
from nltk.tokenize import word_tokenize


class VectorDataset:
    def __init__(self, text_data, y, vectorizer):
        self.text_data = text_data
        self.y = y
        self.vectorizer = vectorizer

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, ind):
        text = self.text_data[ind]
        text = [torch.tensor(self.vectorizer[word]) for word in word_tokenize(text) if word in self.vectorizer.key_to_index]
        if not text:
            return torch.zeros((1, 300)), torch.tensor(self.y[ind])
        return torch.stack(text, axis=0), torch.tensor(self.y[ind])

def data_collator(batch: tuple):
    X = pad_sequence([sample[0] for sample in batch], batch_first=True, padding_side='left')
    y = torch.stack([sample[1] for sample in batch], 0)
    return X, y

def train(model, epochs, optimizer, model_name, train_dataloader, val_dataloader, class_names, scheduler=None, device='cuda'):
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = 0
    losses = []
    f1_scores = []
    for epoch in range(epochs):
        avg_loss = 0
        avg_f1 = 0
        model.train()
        for train_iter, (X, y) in tqdm.tqdm(enumerate(train_dataloader), desc='Training'):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

            avg_loss += loss.item()
        model.eval()
        
        all_preds = []
        for val_iter, (X, y) in tqdm.tqdm(enumerate(val_dataloader), desc='Validation'):
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                logits = model(X)
                
            preds = logits.argmax(-1)
            all_preds.append(preds)
            
        all_preds = torch.cat(all_preds)
        f1 = f1_score(test_y, all_preds.cpu().numpy(), average='macro')

        losses.append(avg_loss / (train_iter + 1))
        f1_scores.append(f1)
        print('Epoch:', epoch + 1, 'Loss:', avg_loss / (train_iter + 1), 'F1:', f1)
        if f1 > best_f1:
            best_f1 = f1
            best_preds = all_preds.cpu().numpy()
            torch.save(model, f'{model_name}.pth')

    epochs = list(range(1, epochs + 1))
    fig, axes = plt.subplots(1, 2, figsize=(25, 5))
    
    axes[0].plot(epochs, losses)
    axes[1].plot(epochs, f1_scores)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro F1')

    print(classification_report(test_y, best_preds, target_names=class_names))