# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
The objective of this experiment is to design and implement a deep learning model that can automatically identify and classify named entities in text using a Bidirectional Long Short-Term Memory (BiLSTM) network. Named Entity Recognition (NER) is a sequence labeling task where each word in a sentence must be assigned a tag indicating whether it belongs to a predefined entity category such as person, organization, location, geopolitical entity, or others.

<img width="346" height="691" alt="Screenshot 2026-02-27 161806" src="https://github.com/user-attachments/assets/d6a4a129-ff2f-467b-a40f-c87421661dfe" />





## DESIGN STEPS

### STEP 1:
Data Preparation
Load the dataset, handle missing values, group words into sentences, create word-to-index and tag-to-index mappings, and convert sentences into numerical sequences. Apply padding so all sequences have equal length.
### STEP 2:
Dataset and DataLoader Creation
Split the data into training and testing sets, define a custom dataset class, and create DataLoader objects to efficiently feed batches of data to the model during training.
### STEP 3:
Model Architecture Design
Build a BiLSTM-based neural network consisting of an embedding layer, bidirectional LSTM layer, and a fully connected output layer that predicts a tag for each word in the sequence.

### STEP 4:
Model Training
Define loss function and optimizer, perform forward pass, compute loss, backpropagate gradients, and update model weights for multiple epochs while tracking training and validation loss.
### STEP 5:
Evaluation and Prediction
Test the trained model on unseen sentences, compare predicted tags with true tags, calculate evaluation metrics, and visualize performance to assess the model’s effectiveness.

## PROGRAM
### Name: Somalaraju Rohini
### Register Number: 212224240156
```
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=256):
        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=word2idx["ENDPAD"])
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        logits = self.fc(x)
        return logits



model = BiLSTMTagger(len(word2idx), len(tag2idx)).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=tag2idx["O"])

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)

            loss = loss_fn(
                outputs.view(-1, outputs.shape[-1]),
                labels.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)

                loss = loss_fn(
                    outputs.view(-1, outputs.shape[-1]),
                    labels.view(-1)
                )

                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses
    print("\nClassification Report:\n")
    print(classification_report(true_tags, pred_tags))


# ==============================
# Inference and prediction
# ==============================

# Find a sentence that actually contains named entities
for idx in range(len(y_test)):
    if any(tag.item() != tag2idx["O"] for tag in y_test[idx]):
        i = idx
        break

model.eval()
sample = X_test[i].unsqueeze(0).to(device)
output = model(sample)
preds = torch.argmax(output, dim=-1).squeeze().cpu().numpy()
true = y_test[i].numpy()

print('Name: Somalaraju Rohini')
print('Register Number: 212224240156')

print(f"\nSentence Index: {i}\n")

print("{:<15} {:<10} {:<10} {}".format("Word", "True", "Pred", "Match"))
print("-"*55)

correct = 0
total = 0
sentence_words = []

for w_id, true_tag, pred_tag in zip(X_test[i], y_test[i], preds):
    if w_id.item() != word2idx["ENDPAD"]:
        word = words[w_id.item() - 1]
        true_label = tags[true_tag.item()]
        pred_label = tags[pred_tag]

        match = "✓" if true_label == pred_label else "✗"

        print(f"{word:<15} {true_label:<10} {pred_label:<10} {match}")

        sentence_words.append(word)

        total += 1
        if true_label == pred_label:
            correct += 1

# Print full sentence
print("\nSentence:")
print(" ".join(sentence_words))

# Sentence accuracy
accuracy = correct / total if total > 0 else 0
print(f"\nSentence Accuracy: {accuracy:.2%}")
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot





<img width="729" height="531" alt="Screenshot 2026-02-27 162156" src="https://github.com/user-attachments/assets/e77d0eb4-fdfe-404b-b4f8-8298ec345ccb" />



<img width="789" height="621" alt="Screenshot 2026-02-27 162228" src="https://github.com/user-attachments/assets/93c62ef2-77f1-4969-a02e-7d706c148a0b" />


### Sample Text Prediction


<img width="1202" height="710" alt="Screenshot 2026-02-27 162311" src="https://github.com/user-attachments/assets/7c88f105-ea9f-482d-a420-918a8ff5f046" />


## RESULT
The BiLSTM-based NER model was successfully trained and evaluated, showing decreasing loss and the ability to correctly predict entity tags for many words in test sentences. Overall, the model demonstrated effective contextual learning and achieved satisfactory performance on the dataset.
