# 用于训练和验证newsflow算法
# 单只划分：有很多股票很多天没有新闻怎么办: ---> 直接eos
# 市场合并：市场整体新闻太多(只用标题？)



# %%
# dataset process for factors (daily)
import os
import json
import zipfile
from glob import glob

def unzip_file(zip_file_path, target_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        zip_ref.extractall(target_dir)
    print(f"文件已成功解压缩到{target_dir}")

# unzip_file("./AI_data/newsflow/news_cn.zip", "./AI_data/newsflow/news_cn/")
# paths = glob("./AI_data/newsflow/news_cn/*.json")
# news = {}
# for path in paths:
#     date = path.split("/")[-1].split(".")[0]
#     with open(path, 'r', encoding='utf-8') as file:
#         news[date] = json.load(file)
# with open("./AI_data/newsflow/news_cn.json", "w", encoding="utf-8") as f:
#     json.dump(news, f)
with open("./AI_data/newsflow/news_cn.json", 'r', encoding='utf-8') as file:
    news_json = json.load(file)
news_json = dict(sorted(news_json.items(), key=lambda item: item[0]))


datas_news_train = {}
datas_news_test = {}
split_cnt = int(len(news_json) * 0.7)
cnt = 0
for k, v in news_json.items():
    news_returns = []
    for st, nv in v.items():
        news_return = []
        if len(nv[0]) == 0:
            news_text = "最近并没有与该股票相关的新闻。"
            print(st, " ", k, " ", news_text)
        else:
            news_text = "\n".join(nv[0])
        news_return = [news_text, nv[1]]
        news_returns.append(news_return)
    if cnt < split_cnt:
        datas_news_train[k] = news_returns
    else:
        datas_news_test[k] = news_returns
    cnt += 1
print("训练集天数：", len(datas_news_train), "\n测试集天数：", len(datas_news_test))

# %%
# model load and LoRA adding

## 读取模型
import torch as th
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
models = ["Meta-Llama-3-8B-Instruct", "Qwen2-7B-Instruct"]
model_path = f'/cpfs/29583eqvgtdvw5cvegg/model/model/{models[1]}'
tokenizer = AutoTokenizer.from_pretrained(model_path)
basemodel = AutoModelForCausalLM.from_pretrained(model_path)

device = th.device('cpu')
basemodel.to(device)
a = basemodel.model.forward(th.tensor([[  40, 3021,  499]]))
print(a.last_hidden_state.shape)
print(a.last_hidden_state)
print("ok")


# %%
# dataset process for news (daily)
from tqdm import tqdm
from typing import Any
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class NewsflowRegressionDataset(Dataset):
    def __init__(self, data, tokenizer, device, max_length=1024 * 8) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.data = []
        with tqdm(total=len(data.keys()), desc="train dataset making") as pbar_outer:
            for _, raws in data.items():
                for raw in raws:
                    text, label = raw   # 这里假设 raw 是个列表，包含文本和预测值
                    encoded_text = self.tokenizer(text, padding=False, truncation=False)
                    self.data.append({
                        'input_ids': th.tensor(encoded_text['input_ids'], dtype=th.long),
                        'label': th.tensor(label, dtype=th.float)
                    })
                pbar_outer.update(1)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Any:
        return self.data[index]


def collate_fn(batch):
    # Extract input_ids and labels from the batch
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Sort the data by sequence length (descending order)
    sorted_batch = sorted(zip(input_ids, labels), key=lambda x: len(x[0]), reverse=True)
    seqs, labels = zip(*sorted_batch)
    
    # Pad the sequences and get their lengths
    seq_lengths = [len(seq) for seq in seqs]
    seq_tensor = pad_sequence(seqs, batch_first=True)
    
    return seq_tensor, th.stack(labels), seq_lengths


# raw_data_train = {
#     '2022-01-01': [["I love you fuck", 0.231]],
#     '2022-01-02': [["I hate you", -0.231]],
# }
raw_data_train = datas_news_train
dataset_train = NewsflowRegressionDataset(raw_data_train, tokenizer, device)
train_dataloader = DataLoader(dataset_train, batch_size=8, shuffle=True, collate_fn=collate_fn)


# raw_data_test = {
#     '2022-01-01': [["I love you", 0.231]],
#     '2022-01-02': [["I hate you", -0.231]],
# }
raw_data_test = datas_news_test
dataset_test = NewsflowRegressionDataset(raw_data_test, tokenizer, device)
test_dataloader = DataLoader(dataset_test, batch_size=8, shuffle=False, collate_fn=collate_fn)


# %%
# Finetune
import pandas as pd
import matplotlib.pyplot as plt

## LoRA Configs
lora_config = LoraConfig(
    r = 4, # Rank
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM
)


class LLM_Finetune:
    def __init__(self, base_model, lora_config, train_dataset, test_dataset, save_path, device, load=False, epoch=10, loss_fn=nn.MSELoss, optimizer=th.optim.Adam):
        self.model = get_peft_model(base_model, lora_config)
        self.dense_net = nn.Linear(base_model.config.hidden_size, 1).to(device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer([
            {'params': self.model.parameters(), 'lr': 1e-4},
            {'params': self.dense_net.parameters(), 'lr': 1e-3}
        ])
        self.epochs = epoch
        self.save_path = save_path
        self.loss_fn = loss_fn()
        self.device = device
        if load:
            self.lora_load()

    def get_last_hidden_state(self, input_ids, attention_mask=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        assert len(input_ids.shape) == 2, ValueError("输入的形状必须是两维的，第一维度为tokens的batch数")
        with th.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]  # 最后一个token对应的hidden state
        return last_hidden_state[:, -1, :]
    
    def get_last_prediction(self, input_ids, attention_mask=None):
        hidden_s = self.get_last_hidden_state(input_ids, attention_mask)
        output = self.dense_net(hidden_s)
        return output
    
    def backtest(self, raw_dataset):
        forecast = []
        for st, datas in raw_dataset.items():
            tmp = {}
            for date, text in datas.items():
                with th.no_grad():
                    encoded_text = tokenizer(text, padding=False, truncation=False)
                    output = self.get_last_prediction(th.tensor(encoded_text['input_ids'], dtype=th.long))
                    tmp[date] = output.squeeze(0).detach().cpu().numpy()[0]
            tmp_df = pd.DataFrame([tmp], index=[st])
            forecast.append(tmp_df)
        df = pd.concat(forecast)
        print(df)
        return df
    
    def test(self, losses_test):
        self.model.eval()
        avg_loss_eval = 0.0
        with th.no_grad():
            for batch in self.test_dataset:
                inputs = batch['input_ids']
                labels = batch['label']
                outputs = self.get_last_prediction(inputs)
                loss = self.loss_fn(outputs, labels)
                avg_loss_eval += loss.item()
                losses_test.append(loss.item())
        print(f"Average Loss for Test: ", avg_loss_eval / len(self.test_dataset))
        return losses_test
    
    def train(self):
        losses_train = []
        losses_test = []
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            avg_loss = 0.0
            for i, (inputs, labels, seq_lengths) in enumerate(self.train_dataset, 0):
                inputs = inputs.to(device)
                labels = labels.to(device)
                packed_inputs = pack_padded_sequence(inputs, seq_lengths, batch_first=True, enforce_sorted=False)
                inputs, _ = pad_packed_sequence(packed_inputs, batch_first=True)

                # 创建 attention mask
                attention_mask = th.zeros_like(inputs)
                for idx, seq_len in enumerate(seq_lengths):
                    attention_mask[idx, :seq_len] = 1  # 标记非 padding 部分

                self.optimizer.zero_grad()

                outputs = self.get_last_prediction(inputs, attention_mask)
                loss = self.loss_fn(outputs.squeeze(), labels)
                loss.backward()
                print(f"Epoch {epoch} loss: ", loss.item())
                avg_loss += loss.item()
                losses_train.append(loss.item())
                self.optimizer.step()
            
            print(f"Average Loss for Train in Epoch {epoch}: ", avg_loss / len(self.train_dataset))
            
            losses_test = self.test(losses_test)
        self.plot(losses_train, losses_test)
        self.lora_save()
        self.lora_load()
                  
    def lora_save(self):
        self.model.save_pretrained(self.save_path + f"model")
        th.save(self.dense_net.state_dict(), self.save_path + "dense_net.pth")
        print("Saving Successfully!")

    def lora_load(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.save_path + f"model")
        self.dense_net.load_state_dict(th.load(self.save_path + "dense_net.pth"))
        print("Loading Successfully!")

    def plot(self, losses_train, losses_test):
        plt.plot(losses_train, color="r", label="training loss")
        plt.plot(losses_test, color="g", label="testing loss")
        plt.legend()
        plt.title("Finetune Losses")
        plt.savefig(self.save_path + "losses.png")
        plt.show()


FinetuneLLM = LLM_Finetune(basemodel, lora_config, train_dataloader, test_dataloader, "./AI_data/newsflow/", device)
print(FinetuneLLM.model.print_trainable_parameters())

hidden_s = FinetuneLLM.get_last_hidden_state(th.tensor([  40, 3021,  499]))
print("LLM hidden state: ", hidden_s)
pred = FinetuneLLM.dense_net(hidden_s)
print("Predition: ", pred)

FinetuneLLM.train()


# %% 
# backtest
raw_dataset = {
    "ADI.O": {"2024-01-01": "what's a pity", "2024-01-02": "happy birthday"},
    "ADP.O": {"2024-01-01": "what happen", "2024-01-02": "what's the hell"},
}
FinetuneLLM.backtest(raw_dataset)