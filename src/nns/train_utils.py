import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
from nltk.tokenize import word_tokenize


class VectorDataset:
    def __init__(self, text_data, y, vectorizer, additional_data=None):
        self.text_data = text_data
        self.y = y
        self.vectorizer = vectorizer
        self.additional_data = additional_data

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, ind):
        text = self.text_data[ind]
        text = [torch.tensor(self.vectorizer[word]) for word in word_tokenize(text) if word in self.vectorizer.key_to_index]
        if not text:
            text = torch.zeros((1, 300))
        else:
            text = torch.stack(text, axis=0)
        if self.additional_data is not None:
            return text, self.additional_data[ind], torch.tensor(self.y[ind])
        return text, torch.tensor(self.y[ind])

def data_collator(batch: tuple):
    X = pad_sequence([sample[0] for sample in batch], batch_first=True, padding_side='left')
    y = torch.stack([sample[-1] for sample in batch], 0)
    
    if len(batch[0]) == 3:
        add_data = torch.stack([sample[1] for sample in batch], 0)
        return X, add_data, y
        
    return X, y

def batch_to_device(batch: tuple, device):
    return tuple(map(lambda x: x.to(device), batch))

def train(model, epochs, optimizer, model_name, train_dataloader, val_dataloader, class_names, scheduler=None, device='cuda'):
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = 0
    losses = []
    f1_scores = []
    for epoch in range(epochs):
        avg_loss = 0
        avg_f1 = 0
        model.train()
        for train_iter, batch in tqdm.tqdm(enumerate(train_dataloader), desc='Training'):
            batch = batch_to_device(batch, device)
            X, y = batch[:-1], batch[-1]
            logits = model(*X)
            loss = loss_fn(logits, y)
            loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

            avg_loss += loss.item()
        model.eval()
        
        all_preds = []
        all_labels = []
        for val_iter, batch in tqdm.tqdm(enumerate(val_dataloader), desc='Validation'):
            batch = batch_to_device(batch, device)
            X, y = batch[:-1], batch[-1]
            with torch.no_grad():
                logits = model(*X)
                
            preds = logits.argmax(-1)
            all_preds.append(preds)
            all_labels.append(y)
            
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        f1 = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')

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

    print(classification_report(all_labels.cpu().numpy(), best_preds, target_names=class_names))