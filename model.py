import os
import ssl

# 禁用SSL验证（仅用于开发环境）
ssl._create_default_https_context = ssl._create_unverified_context

# 设置Hugging Face镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import BertModel, BertConfig
import torch.nn as nn

class BertMultiTask(nn.Module):
    def __init__(self, num_labels1, num_labels2, num_labels3, model_name='bert-base-chinese'):
        super(BertMultiTask, self).__init__()
        # 加载预训练的Bert模型
        self.bert = BertModel.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        
        # 获取Bert的隐藏层维度
        self.hidden_size = self.bert.config.hidden_size
        
        # 三个分类头
        self.classifier1 = nn.Linear(self.hidden_size, num_labels1)
        self.classifier2 = nn.Linear(self.hidden_size, num_labels2)
        self.classifier3 = nn.Linear(self.hidden_size, num_labels3)
    
    def forward(self, input_ids, attention_mask):
        # 获取Bert的输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用[CLS] token的表示作为句子表示
        cls_output = outputs.hidden_states[-1][:, 0, :]
        
        # 三个分类任务的输出
        logits1 = self.classifier1(cls_output)
        logits2 = self.classifier2(cls_output)
        logits3 = self.classifier3(cls_output)
        
        return logits1, logits2, logits3
