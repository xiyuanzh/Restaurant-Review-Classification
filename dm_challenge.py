from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


# Function to calculate the accuracy of our predictions vs labels
def compute_metrics(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten().tolist()
    labels_flat = labels.flatten().tolist()
    acc = accuracy_score(labels_flat, pred_flat)
    macro = f1_score(labels_flat, pred_flat, average='macro')
    micro = f1_score(labels_flat, pred_flat, average='micro')
    return acc, macro, micro

def preprocess_df(df):
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    preprocessed_sentences = []
    for i, row in df.iterrows():
        sent = row["text"]
        sent_nopuncts = sent.translate(translator)
        words_list = sent_nopuncts.strip().split()
        filtered_words = [word for word in words_list if word not in stop_words and len(word) != 1]
        preprocessed_sentences.append(" ".join(filtered_words))
    df["text"] = preprocessed_sentences
    return df

data_path = "./"

df_train = pd.read_csv(data_path + "train.csv")
df_test = pd.read_csv(data_path + "test.csv")

labels = []
label_dict = dict()
counter = 0
for l in df_train['label']:
    if not l in label_dict:
        label_dict[l] = counter
        counter += 1
    labels.append(label_dict[l])
print(label_dict)
num_labels = len(label_dict)

df_train['text'] = df_train['name'] + df_train['review']
df_test['text'] = df_test['name'] + df_test['review']
df_train = preprocess_df(df_train)
df_test = preprocess_df(df_test)

print('tokenizing...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

input_ids = []
attention_masks = []

docs = df_train['text']

for doc in docs:
    encoded_dict = tokenizer.encode_plus(
        doc,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
        #padding='max_length',
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

print('tokenizing test...')

test_input_ids = []
test_attention_masks = []

test_docs = df_test['text']

for doc in test_docs:
    test_encoded_dict = tokenizer.encode_plus(
        doc,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
        #padding='max_length',
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    test_input_ids.append(test_encoded_dict['input_ids'])
    test_attention_masks.append(test_encoded_dict['attention_mask'])

test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)

train_dataset = TensorDataset(input_ids, attention_masks, labels)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    output_attentions=False,
    output_hidden_states=False,
)

model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

epochs = 5
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

print('training...')
best_val = 0
for epoch_i in range(0, epochs):

    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()
        loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    print("Total training loss: {0:.4f}".format(total_train_loss))

    model.eval()
    torch.save(model.state_dict(), "best" + str(epoch_i))

label2id = dict()
label_key_list = list(label_dict.keys())
for i in range(len(label_key_list)):
    label2id[i] = label_key_list[i]

test_dataset = TensorDataset(test_input_ids, test_attention_masks)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

model.eval()

for k in range(5):
    dic = {"Id": [], "Predicted": []}
    model.load_state_dict(torch.load("best" + str(k)))

    for niter, batch in enumerate(test_dataloader):

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = logits[0].cpu().numpy()
        pred_flat = np.argmax(logits, axis=1).flatten().tolist()

        for i, pred in enumerate(pred_flat):
            dic["Id"].append(niter * batch_size + i)
            dic["Predicted"].append(label2id[pred_flat[i]])

    dic_df = pd.DataFrame.from_dict(dic)
    print(data_path + "predicted" + str(k) + ".csv")
    dic_df.to_csv(data_path + "predicted" + str(k) + ".csv", index=False)