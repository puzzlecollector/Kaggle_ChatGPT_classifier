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
import seaborn as sns 

df_gpt = pd.read_csv("wiki_batch_1_gpt.csv") 
df_human = pd.read_csv("wiki_batch_1_human.csv") 
df = pd.concat([df_human, df_gpt], axis=0) 
df = df.sample(frac=1).reset_index(drop=True) # shuffle rows 

input_ids, attn_masks = [], [] 

titles = df["titles"].values 
contents = df["contents"].values 
labels = df["labels"].values 

tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base") 


for i in tqdm(range(len(titles)), position=0, leave=True): 
    encoded_input = tokenizer(str(titles[i]), str(contents[i]), max_length=512, truncation=True, padding="max_length") 
    input_ids.append(encoded_input["input_ids"]) 
    attn_masks.append(encoded_input["attention_mask"]) 
    

input_ids = torch.tensor(input_ids, dtype=int) 
attn_masks = torch.tensor(attn_masks, dtype=int) 
labels = torch.tensor(labels, dtype=int) 

# train/validation split in 8:2 ratio 
train_size = int(input_ids.shape[0] * 0.8) 

train_input_ids = input_ids[:train_size] 
train_attn_masks = attn_masks[:train_size] 
train_labels = labels[:train_size] 

val_input_ids = input_ids[train_size:] 
val_attn_masks = attn_masks[train_size:] 
val_labels = labels[train_size:] 

print(train_input_ids.shape, train_attn_masks.shape, train_labels.shape, val_input_ids.shape, val_attn_masks.shape, val_labels.shape) 


# create dataloaders 
batch_size = 32 

train_data = TensorDataset(train_input_ids, train_attn_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size) 
    
val_data = TensorDataset(val_input_ids, val_attn_masks, val_labels) 
val_sampler = SequentialSampler(val_data) 
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

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
        self.config = AutoConfig.from_pretrained("monologg/kobigbird-bert-base")
        self.lm = AutoModel.from_pretrained("monologg/kobigbird-bert-base") 
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

def flat_accuracy(preds, labels): 
    pred_flat = np.argmax(preds, axis=1).flatten() 
    labels_flat = labels.flatten() 
    return np.sum(pred_flat == labels_flat) / len(labels_flat) 

# no class weights are needed as the classes are almost perfectly balanced 
best_val_loss = 99999999999


device = torch.device("cuda") 
loss_func = nn.CrossEntropyLoss() 
model = ChatGPTDetector().to(device) 
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8) 
epochs = 10
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = int(0.05*total_steps), 
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
    for step, batch in tqdm(enumerate(val_dataloader), position=0, leave=True, desc="Validating", total=len(val_dataloader)): 
        batch = tuple(t.to(device) for t in batch) 
        b_input_ids, b_attn_masks, b_labels = batch 
        with torch.no_grad(): 
            outputs = model(b_input_ids, b_attn_masks) 
            loss = loss_func(outputs, b_labels) 
            val_loss += loss.item() 
            val_accuracy += flat_accuracy(outputs.detach().cpu().numpy(), b_labels.detach().cpu().numpy()) 
    avg_val_loss = val_loss / len(val_dataloader) 
    avg_val_accuracy = val_accuracy / len(val_dataloader) 
    
    print(f"train loss:{avg_train_loss} | train accuracy:{avg_train_accuracy} | val loss:{avg_val_loss} | val accuracy:{avg_val_accuracy}")
    
    if best_val_loss > avg_val_loss: 
        best_val_loss = avg_val_loss 
        torch.save(model.state_dict(), "KR_ChatGPT_Detector_v1_.pt") 
    
print("done training!") 
print(f"best validation loss : {best_val_loss}")
