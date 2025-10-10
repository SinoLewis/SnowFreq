import torch.nn as nn
from transformers import BertModel, DistilBertModel, AutoModel, AlbertModel
import torch
from torch.utils.data import Dataset

#   * **DistilBERT** (`distilbert-base-uncased`, 66M params, \~40% faster, \~97% accuracy of BERT).
#   * **MiniLM** (`microsoft/MiniLM-L6-H384-uncased`), \~22M params, much faster.
#   * **ALBERT** (`albert-base-v2`), parameter sharing reduces size.

# ----------------- 3. Fusion Model --------------------
class SentimentFusionModel(nn.Module):
    def __init__(self, tfidf_dim, hidden_dim=256, num_classes=3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # self.bert = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        # self.bert = AlbertModel.from_pretrained("albert-base-v2")
        # 🔒 Freeze all BERT parameters (no gradient updates)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.tfidf_proj = nn.Sequential(
            nn.Linear(tfidf_dim, 128),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),  # multi-class classification
        )

    def forward(self, input_ids, attention_mask, tfidf):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]
        tfidf_out = self.tfidf_proj(tfidf)
        fused = torch.cat([cls_embedding, tfidf_out], dim=1)
        return self.classifier(fused)

# ----------------- 2. Dataset Class --------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, tfidf_vectorizer, label_map):
        self.labels = [label_map[label] for label in labels]
        self.tokenizer = tokenizer
        self.tfidf_vectors = tfidf_vectorizer.transform(texts).astype("float32")  # keep CPU-side, float32
        self.bert_encodings = tokenizer(
            texts.tolist(),
            padding=True, truncation=True, max_length=128,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.bert_encodings["input_ids"][idx],
            "attention_mask": self.bert_encodings["attention_mask"][idx],
            "tfidf": torch.from_numpy(self.tfidf_vectors[idx].toarray().squeeze()),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }