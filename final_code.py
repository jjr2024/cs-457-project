from sklearn.model_selection import train_test_split
import numpy as np
import requests
import json
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel
from joblib import dump, load
from sklearn.model_selection import train_test_split

"""
MAIN ANALYSIS
"""








"""
GOOGLE ANALYSIS
"""
df_data = pd.read_csv("GOOG_transcripts.csv")

df_data.columns = ["text", "label"]
df_data = df_data.dropna()
X = df_data.drop(columns=['label'])
y = df_data['label']

# Split the data into train, dev, and test sets (80-10-10 split)
X_train, X_devtest, y_train, y_devtest = train_test_split(X, y, test_size=0.2)

X_dev, X_test, y_dev, y_test = train_test_split(X_devtest, y_devtest, test_size=0.5)

df_train = X_train.join(y_train)
df_dev = X_dev.join(y_dev)
df_test = X_test.join(y_test)

#All key inputs
num_labels = 3  # Number of labels (neutral 0, bad 1, good 2)
MAX_LENGTH = 128
batch_size = 5  # Number for minibatch training here
num_epochs = 20 # Number of training epochs

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the device we want to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
model.to(device)  # Move the model to the right device

def data_preprocess(df):
    # Preprocessing the data
    # Tokenize the text data
    tokenized_texts = []
    labels = []
    for i, row in df.iterrows():
        tokenized_text = tokenizer.encode(row['text'], add_special_tokens=True, max_length=512, truncation=True)
        tokenized_texts.append(tokenized_text)
        labels.append(row['label'])

    # Define the label mapping
    label_map = {0: 0, -1: 1, 1: 2}

    # Change labels to be consistent with label mapping above
    labels = [label_map[label] for label in labels]

    input_ids = torch.tensor([tokenized_text[:MAX_LENGTH] + [0] * (MAX_LENGTH - len(tokenized_text[:MAX_LENGTH])) for tokenized_text in tokenized_texts])
    labels = torch.tensor(labels)
    # Output data that's ready to be put into a dataloader
    data = TensorDataset(input_ids, labels)
    return data

# Training loop
loss_vec = []
def train_model(lr, batch_size, num_epochs, data):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr) # Define the optimizer
    avg_train_loss = None
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, labels = batch

            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss}")
    loss_vec.append(avg_train_loss)

# Prediction function
def predict(text):
    # Tokenize text
    tokenized_text = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
    
    # Convert tokenized input to tensor and move it to the device
    input_ids = torch.tensor(tokenized_text).unsqueeze(0).to(device)
    
    # Set the model to eval mode
    model.eval()
    
    with torch.no_grad():
        # Give model the inputs
        outputs = model(input_ids)
        
        # Get the logits from the model's output
        logits = outputs.logits
    
    # Calculate the probabilities using softmax
    probabilities = torch.softmax(logits, dim=-1).squeeze(0)
    
    # Get the predicted label
    predicted_label = torch.argmax(probabilities).item()
    
    # Return the predicted label and probabilities
    return probabilities, predicted_label

# Below is our dev loop: finding the best hyperparameters to use for the training loop
data = data_preprocess(df_dev)
lr_vec = []
batch_size_vec = []
for i in range(1,6):
    print(f"Learning Rate: {i*1e-5}")
    for j in range (1,4):
        print(f"Batch Size: {j*16}")
        lr_vec.append(i*1e-5)
        batch_size_vec.append(j*16)
        train_model(lr=i*1e-5, batch_size=j*16, num_epochs=3, data=data)
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# We find the minimum final epoch loss. Then we set best_lr and best_bsize to the learning rate and batch sizes that correspond to that minimum loss
best_lr = lr_vec[loss_vec.index(min(loss_vec))]
best_bsize = batch_size_vec[loss_vec.index(min(loss_vec))]

# Training the model
print(f"Lowest loss, best learning rate, batch size {min(loss_vec), best_lr, best_bsize}")
data = data_preprocess(df_train)
train_model(lr=best_lr, batch_size=best_bsize, num_epochs=10, data=data)


############################################
# The below section is dedicated to testing our data and outputting accuracy scores and label distributions

# Initialize lists to store predictions
predictions_label0 = []
predictions_label1 = []
predictions_label2 = []
predictions_predict = []

# getBERTScores adds "neutral," "bad," "good," and "predict" features to the
# argument dataframe. The first three new features correspond to probabilities for each label,
# while the "predict" feature is the predicted label 
def getBERTScores(df): 
    predictions_label0 = []
    predictions_label1 = []
    predictions_label2 = []
    predictions_predict = []

    for _, row in df.iterrows():
        predictions = predict(row["text"])
        
        # Append the prediction to the list
        predictions_label0.append(predictions[0][0])
        predictions_label1.append(predictions[0][1])
        predictions_label2.append(predictions[0][2])
        predictions_predict.append(predictions[1])

    # Add the predictions as a new feature to X_test
    rev_label_map = {0: 0, 1: -1, 2: 1}
    predictions_predict = [rev_label_map[label] for label in predictions_predict]

    df['neutral'] = predictions_label0
    df['bad'] = predictions_label1
    df['good'] = predictions_label2
    df["predict"] = predictions_predict

getBERTScores(X_test)
getBERTScores(X_train)

#Find our overall accuracy
print(f"Training Accuracy: {(X_train['predict'] == y_train).mean()}")
print(f"Testing Accuracy: {(X_test['predict'] == y_test).mean()}")

#The below code shows our benchmarks
most_frequent_items = y_train.value_counts()
most_frequent = most_frequent_items.head(1)
print(f"Benchmark for Train Set {most_frequent/len(y_train)}")
most_frequent_items = y_test.value_counts()
most_frequent = most_frequent_items.head(1)
print(f"Benchmark for Test Set {most_frequent/len(y_test)}")

#The below print statements show the label distributions of our train and test sets
print("Proportion of test data actually labeled 0:", y_test[y_test == 0].shape[0] / y_test.shape[0])
print("Proportion of test data actually labeled 1:", y_test[y_test == 1].shape[0] / y_test.shape[0])
print("Proportion of test data actually labeled -1:", y_test[y_test == -1].shape[0] / y_test.shape[0])
print(f"Number of test observations: {y_test.shape[0]}")

print("Proportion of training data actually labeled 0:", y_train[y_train == 0].shape[0] / y_train.shape[0])
print("Proportion of training data actually labeled 1:", y_train[y_train == 1].shape[0] / y_train.shape[0])
print("Proportion of training data actually labeled -1:", y_train[y_train == -1].shape[0] / y_train.shape[0])
print(f"Number of training observations: {y_train.shape[0]}")

train_predict = X_train['predict']
test_predict = X_test['predict']

#The below print statements show the label distributions of our predictions
print("Proportion of test data predicted to be labeled 0:", test_predict[test_predict == 0].shape[0] / test_predict.shape[0])
print("Proportion of test data predicted to be labeled 1:", test_predict[test_predict == 1].shape[0] / test_predict.shape[0])
print("Proportion of test data predicted to be labeled -1:", test_predict[test_predict == -1].shape[0] / test_predict.shape[0])
print(f"Number of test observations: {test_predict.shape[0]}")

print("Proportion of training data predicted to be labeled 0:", train_predict[train_predict == 0].shape[0] / train_predict.shape[0])
print("Proportion of training data predicted to be labeled 1:", train_predict[train_predict == 1].shape[0] / train_predict.shape[0])
print("Proportion of training data predicted to be labeled -1:", train_predict[train_predict == -1].shape[0] / train_predict.shape[0])
print(f"Number of training observations: {train_predict.shape[0]}")


############################################
# Below, we test our BERT model on Apple earnings transcript data and output the results

df_AAPL = pd.read_csv("AAPL_transcripts.csv")
df_AAPL.columns = ["text", "label"]
df_AAPL = df_AAPL.dropna()

getBERTScores(df_AAPL)

AAPL_predict = df_AAPL['predict']

print("Proportion of AAPL data labeled 0:", AAPL_predict[AAPL_predict == 0].shape[0] / AAPL_predict.shape[0])
print("Proportion of AAPL data labeled 1:", AAPL_predict[AAPL_predict == 1].shape[0] / AAPL_predict.shape[0])
print("Proportion of AAPL data labeled -1:", AAPL_predict[AAPL_predict == -1].shape[0] / AAPL_predict.shape[0])
print(f"Number of AAPL observations: {AAPL_predict.shape[0]}")

print(f"AAPL Accuracy: {(df_AAPL['predict'] == df_AAPL['label']).mean()}")

most_frequent_items = df_AAPL['label'].value_counts()
most_frequent = most_frequent_items.head(1)
print(f"Benchmark for Test Set {most_frequent/len(df_AAPL['label'])}")


############################################
# Below, we test our BERT model on Microsoft earnings transcript data and output the results

df_MSFT = pd.read_csv("MSFT_transcripts.csv")
df_MSFT.columns = ["text", "label"]
df_MSFT = df_MSFT.dropna()

getBERTScores(df_MSFT)

MSFT_predict = df_MSFT['predict']

print("Proportion of MSFT data with predicted label 0:", MSFT_predict[MSFT_predict == 0].shape[0] / MSFT_predict.shape[0])
print("Proportion of MSFT data with predicted label 1:", MSFT_predict[MSFT_predict == 1].shape[0] / MSFT_predict.shape[0])
print("Proportion of MSFT data with predicted label -1:", MSFT_predict[MSFT_predict == -1].shape[0] / MSFT_predict.shape[0])
print(f"Number of MSFT observations: {MSFT_predict.shape[0]}")

print(f"MSFT Accuracy: {(df_MSFT['predict'] == df_MSFT['label']).mean()}")

most_frequent_items = df_MSFT['label'].value_counts()
most_frequent = most_frequent_items.head(1)
print(f"Benchmark for Test Set {most_frequent/len(df_MSFT['label'])}")

print("Proportion of MSFT data actually labeled 0:", df_MSFT[df_MSFT["label"] == 0].shape[0] / df_MSFT.shape[0])
print("Proportion of MSFT data actually labeled 1:", df_MSFT[df_MSFT["label"] == 1].shape[0] / df_MSFT.shape[0])
print("Proportion of MSFT data actually labeled -1:", df_MSFT[df_MSFT["label"] == -1].shape[0] / df_MSFT.shape[0])
print(df_MSFT.shape[0])