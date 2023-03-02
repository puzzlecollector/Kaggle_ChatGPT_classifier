import numpy as np 
import pandas as pd 
import os 
from transformers import *
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler 
import time
from datetime import datetime 
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
import torch
import torch.nn as nn 
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

INF = 999999999999999

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large") 


df = pd.read_csv("GPT-wiki-intro.csv") 

questions = df["prompt"].values 
human_answers = df["wiki_intro"].values 
chatgpt_answers = df["generated_text"].values 
prompts, answers, labels = [], [], []

for i in tqdm(range(len(questions))):
    for j in range(len(human_answers[i])):
        prompts.append(questions[i]) 
        answers.append(human_answers[i][j]) 
        labels.append(0) 
    for j in range(len(chatgpt_answers[i])): 
        prompts.append(questions[i]) 
        answers.append(chatgpt_answers[i][j]) 
        labels.append(1) 
    
wiki_df = pd.DataFrame(list(zip(prompts, answers, labels)), columns=["prompt", "answer", "AI"])

print(wiki_df.head()) 

input_ids, attn_masks = [], [] 

prompts = wiki_df["prompt"].values 
answers = wiki_df["answer"].values 
labels = wiki_df["AI"].values

for i in tqdm(range(len(prompts)), position=0, leave=True): 
    encoded_input = tokenizer(str(prompts[i]), str(answers[i]), max_length=512, truncation=True, padding="max_length") 
    input_ids.append(encoded_input["input_ids"])
    attn_masks.append(encoded_input["attention_mask"]) 

input_ids = torch.tensor(input_ids, dtype=int) 
attn_masks = torch.tensor(attn_masks, dtype=int) 
labels = torch.tensor(labels, dtype=int) 

def flat_accuracy(preds, labels): 
    pred_flat = np.argmax(preds, axis=1).flatten() 
    labels_flat = labels.flatten() 
    return np.sum(pred_flat == labels_flat) / len(labels_flat) 



class MeanPooling(nn.Module): 
    def __init__(self): 
        super(MeanPooling, self).__init__() 
    def forward(self, last_hidden_state, attention_masks): 
        input_mask_expanded = attention_masks.unsqueeze(-1).expand(last_hidden_state.size()).float() 
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1) 
        sum_mask = input_mask_expanded.sum(1) 
        sum_mask = torch.clamp(sum_mask, min=1e-9) 
        mean_embeddings = sum_embeddings / sum_mask 
        return mean_embeddings 
    
class MultiSampleDropout(nn.Module): 
    def __init__(self, max_dropout_rate, num_samples, classifier):
        super(MultiSampleDropout, self).__init__()
        self.dropout = nn.Dropout
        self.classifier = classifier
        self.max_dropout_rate = max_dropout_rate
        self.num_samples = num_samples
    def forward(self, out):
        return torch.mean(torch.stack([self.classifier(self.dropout(p=rate)(out)) for _, rate in enumerate(np.linspace(0, self.max_dropout_rate, self.num_samples))], dim=0), dim=0)

class ChatGPTDetector(nn.Module): 
    def __init__(self, num_classes=2): 
        super(ChatGPTDetector, self).__init__() 
        self.num_classes = num_classes 
        self.config = AutoConfig.from_pretrained("microsoft/deberta-v3-large")
        self.lm = AutoModel.from_pretrained("microsoft/deberta-v3-large") 
        self.mean_pooler = MeanPooling() 
        self.fc = nn.Linear(self.config.hidden_size, self.num_classes) 
        self._init_weights(self.fc) 
        self.multi_dropout = MultiSampleDropout(0.2, 8, self.fc) 
    def _init_weights(self, module):
        if isinstance(module, nn.Linear): 
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range) 
            if module.bias is not None: 
                module.bias.data.zero_() 
    def forward(self, input_ids, attn_masks):
        x = self.lm(input_ids, attn_masks)[0]
        x = self.mean_pooler(x, attn_masks) 
        x = self.multi_dropout(x)
        return x 
    
    
kf = StratifiedKFold(n_splits=5) 

for idx, (train_idx, val_idx) in enumerate(kf.split(input_ids, labels)): 
    if idx != 0:
        continue 
    print(f"======== KFOLD {idx+1} ========")
    train_input_ids, val_input_ids = input_ids[train_idx], input_ids[val_idx]
    train_attn_masks, val_attn_masks = attn_masks[train_idx], attn_masks[val_idx] 
    train_labels, val_labels = labels[train_idx], labels[val_idx] 
    
    print(train_labels.shape, val_labels.shape) 
    
    batch_size = 24
    train_data = TensorDataset(train_input_ids, train_attn_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size) 
    
    val_data = TensorDataset(val_input_ids, val_attn_masks, val_labels) 
    val_sampler = SequentialSampler(val_data) 
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size) 
    
    class_weights = compute_class_weight(class_weight="balanced", classes=torch.unique(train_labels).numpy(), y=train_labels.numpy())
    class_weights = torch.tensor(class_weights).float().to(device) 
    loss_func = nn.CrossEntropyLoss(weight=class_weights) 
    
    best_val_loss = INF 
    
    model = ChatGPTDetector() 
    hc3_chkpt = torch.load("DeBERTaLarge_HC3_1.pt") 
    print(model.load_state_dict(hc3_chkpt)) 
    model.cuda()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8) 
    epochs = 5
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = int(0.01*total_steps), 
                                                num_training_steps = total_steps) 
    model.zero_grad()
    for epoch_i in tqdm(range(0, epochs), desc="Epochs", position=0, leave=True, total=epochs):
        train_loss, train_accuracy = 0, 0   
        model.train() 
        with tqdm(train_dataloader, unit="batch") as tepoch: 
            for step, batch in enumerate(tepoch): 
                batch = tuple(t.to(device) for t in batch) 
                b_input_ids, b_attn_masks, b_labels = batch 
                outputs = model(b_input_ids, b_attn_masks) 
                loss = loss_func(outputs, b_labels) 
                train_loss += loss.item() 
                train_accuracy += flat_accuracy(outputs.detach().cpu().numpy(), b_labels.detach().cpu().numpy())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step() 
                model.zero_grad() 
                tepoch.set_postfix(loss=train_loss/(step+1), accuracy=train_accuracy/(step+1)) 
                time.sleep(0.1) 
        avg_train_loss = train_loss / len(train_dataloader) 
        avg_train_accuracy = train_accuracy / len(train_dataloader) 
        
        val_loss, val_accuracy = 0, 0 
        model.eval() 
        for step, batch in tqdm(enumerate(val_dataloader), position=0, leave=True, total=len(val_dataloader)): 
            batch = tuple(t.to(device) for t in batch) 
            b_input_ids, b_attn_masks, b_labels = batch 
            with torch.no_grad(): 
                outputs = model(b_input_ids, b_attn_masks) 
                loss = loss_func(outputs, b_labels)
                val_loss += loss.item() 
                val_accuracy += flat_accuracy(outputs.detach().cpu().numpy(), b_labels.detach().cpu().numpy())
        avg_val_loss = val_loss / len(val_dataloader) 
        avg_val_accuracy = val_accuracy / len(val_dataloader) 
        
        print(f"avg train loss : {avg_train_loss} | avg train accuracy : {avg_train_accuracy} | avg val loss : {avg_val_loss} | avg val accuracy : {avg_val_accuracy}")
        
        if best_val_loss > avg_val_loss: 
            best_val_loss = avg_val_loss 
            torch.save(model.state_dict(), f"DeBERTaLarge_Wiki_{idx+1}.pt")
            
    print("Done!") 
    print(f"Best validation loss : {best_val_loss}")
    
