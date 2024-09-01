import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F


st.title('うつ病分析')
st.caption('これはＳＮＳの投稿からうつ病かどうかを判定するやつです')


text = st.text_input('分析する文章')

model_name = 'tohoku-nlp/bert-base-japanese-v2'
model = BertForSequenceClassification.from_pretrained(model_name,num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_name)

model_path = './bert_senchmental_model.pth'
model.load_state_dict(torch.load(model_path))

model.eval()

encoding = tokenizer(text,truncation = True,padding = True,max_length=512,return_tensors = 'pt')
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

# モデルによる予測
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=-1)
    logits = outputs.logits

predicted_label = predictions.item()

# ラベルの解釈
label_map = {0: "正常", 1: "鬱傾向あり"}  # 例: 0=ネガティブ, 1=ポジティブ
predicted_label_str = label_map[predicted_label]

probabilities = F.softmax(logits,dim=-1)
predicted_probs  = probabilities.squeeze().tolist()

st.text(f"入力 : {text}")
st.text(f"判定 : {predicted_label_str}")
st.text(f"確率 : [正常],[鬱傾向]: {predicted_probs}")
