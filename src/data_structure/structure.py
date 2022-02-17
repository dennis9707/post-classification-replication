# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import torch


class QuestionDataset(Dataset):

    def __init__(self, df, tokenizer, label):
        self.qid = df['id']
        self.title = df['title']
        self.body = df['body']
        self.label = df[label]
        self.tokenizer = tokenizer
        self.length = 0

    def __len__(self):
        return len(self.qid)

    def __getitem__(self, index):
        title = str(self.title[index])
        body = str(self.body[index])
        tokens = title + body
        
        inputs = self.tokenizer(
            tokens,
            None,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        tags = self.label[index]
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'mask': inputs['attention_mask'].flatten(),
            'labels': torch.from_numpy(tags).type(torch.FloatTensor)
        }


