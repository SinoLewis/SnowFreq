from Features import load_data
from model import SentimentFusionModel, NewsDataset
from transformers import BertTokenizer, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, classification_report
from torch.cuda.amp import GradScaler, autocast
import os

class LLMApp:
    def __init__(self, model_type="bert", model_name="distilbert-base-uncased"):
        # --- Data ---
        self.tfidf_vectorizer = None
        self.train_loader = None
        self.eval_loader = None
        self.test_loader = None

        # --- Models ---
        self.model_type = model_type          # "tfidf_logreg", "tfidf_ann", "bert_ann", "fusion"
        self.model_name = model_name          # e.g. DistilBERT, MiniLM, ALBERT
        self.model = None

        # --- Training ---
        self.optimizer = None
        self.loss_fn = None
        self.trainer = None

        # --- Device ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

    def preprocess(self, train, val, test, batch_size=32):
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_vectorizer.fit(train['content'])

        train_dataset = NewsDataset(train['content'], train['sentiment_class'], tokenizer, self.tfidf_vectorizer, label_map)
        val_dataset   = NewsDataset(val['content'], val['sentiment_class'], tokenizer, self.tfidf_vectorizer, label_map)
        test_dataset  = NewsDataset(test['content'], test['sentiment_class'], tokenizer, self.tfidf_vectorizer, label_map)

        use_pin_memory = True if self.device.type == "cuda" else False

        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=use_pin_memory)
        self.val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=use_pin_memory)
        self.test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, pin_memory=use_pin_memory)

    # -----------------------
    # Initialize Models
    # -----------------------
    def init_model(self):
        # build selected model (TFIDF-LogReg, TFIDF-ANN, BERT-ANN, Fusion)
        self.model = SentimentFusionModel(tfidf_dim=self.tfidf_vectorizer.get_feature_names_out().shape[0])
    # -----------------------
    # Train + Evaluate
    # -----------------------
    def train(self, epochs=5):
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()
        for epoch in range(epochs):
            # ---------- Training ----------
            self.model.train()
            for batch in self.train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                tfidf = batch['tfidf'].to(self.device)
                labels = batch['label'].long().to(self.device)

                # ⚡ Use autocast to run forward in mixed precision
                with autocast():
                    outputs = self.model(input_ids, attention_mask, tfidf)
                    loss = criterion(outputs, labels)

                # Scale loss for stability
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

    def metrics(self, labels, preds, probs):
        acc_val = accuracy_score(labels, preds)
        prec_val = precision_score(labels, preds, average="weighted")
        f1_val = f1_score(labels, preds, average="weighted")
        auc_val = roc_auc_score(labels, probs, multi_class="ovr")

        # print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train - Acc: {acc_val:.4f} | Prec: {prec_val:.4f} | F1: {f1_val:.4f} | AUC: {auc_val:.4f}")

        return acc_val, prec_val, f1_val, auc_val
    def evaluate(self):
        val_loss = 0
        val_loss_all, val_preds, val_probs, val_labels = [], [], [], []
        criterion = nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch in self.eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                tfidf = batch['tfidf'].to(self.device)
                labels = batch['label'].long().to(self.device)

                outputs = self.model(input_ids, attention_mask, tfidf)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probabilities = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probabilities, dim=1)

                val_preds.extend(preds.detach().cpu().numpy())
                val_probs.extend(probabilities.detach().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_loss_all.append(val_loss / len(self.eval_loader))
        print(f"  Train - Loss: {val_loss_all[-1]:.4f}")
        print("Evaluation Metrics")
        mets = self.metrics(val_labels, val_preds, val_probs)
        results = {
            "val_loss_all": val_loss_all,   # list of validation losses per epoch
            "metrics": mets,                # dict of metrics (acc, f1, prec, auc)
            "val_labels": val_labels,       # true labels
            "val_preds": val_preds,         # predicted labels
            "val_probs": val_probs          # predicted probabilities
        }

        return results
    def predict(self):
        """
        Run prediction on the test set using the trained model.
        Returns:
            dict: containing predictions, probabilities, and true labels.
        """
        self.model.eval()  # switch to evaluation mode
        preds, probs, labels = [], [], []

        with torch.no_grad():
            for batch in self.test_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                tfidf = batch['tfidf'].to(self.device)
                label = batch['label'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids, attention_mask, tfidf)

                # Get probabilities and predicted class
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                # Store results
                preds.extend(predictions.cpu().numpy())
                probs.extend(probabilities.cpu().numpy())
                labels.extend(label.cpu().numpy())

        results = {
            "preds": preds,      # predicted class indices
            "probs": probs,      # prediction probabilities
            "labels": labels     # ground truth labels
        }
        return results
    # -----------------------
    # Utilities
    # -----------------------
    def save_model(self,  model_name, path="models/"):
        os.makedirs(path, exist_ok=True)

        # Save TF-IDF vectorizer
        # joblib.dump(tfidf_vectorizer, os.path.join(path, "tfidf_vectorizer.pkl"))

        # Save ANN fusion model (PyTorch)
        torch.save(self.model.state_dict(), os.path.join(path, model_name))

        print(f"[✅] Sentiment Analysis model & TF-IDF saved to {path}")

    def load_model(self, path, model_class, **kwargs):
        """
        Load a saved model checkpoint.
        
        Args:
            path (str): Path to the saved model .pt or .bin file.
            model_class (nn.Module): The class used to define the model.
            **kwargs: Any arguments needed to reinitialize the model class.

        Returns:
            nn.Module: Loaded model ready for inference or training.
        """
        # Rebuild architecture
        self.model = model_class(**kwargs)

        # Load weights
        self.model.load_state_dict(torch.load(path, map_location=self.device))

        # Send model to device
        self.model.to(self.device)

        print(f"✅ Model loaded from {path} onto {self.device}")
        return self.model

    def main(self):
        # Load dataset and create train/eval/test splits
        train, val, test = load_data('../data/Kaggle/cryptonews.csv', test_size=0.2, val_size=0.1)

        # Run preprocessing (TFIDF vectorization or BERT tokenization depending on model_type)
        self.preprocess(train, val, test)

        self.init_model()

        self.train(epochs=5)

        results = self.evaluate()
        print("Evaluation results:", results)

        predictions = self.predict()
        print("Predictions results:", predictions)   # e.g., ["positive", "negative"]

        # Save
        self.save_model(model_name = "fusion_model.pt")

        tfidf_dim = len(self.tfidf_vectorizer.get_feature_names_out())

        # Load later
        self.load_model(
            path='models/',
            model_class=SentimentFusionModel,
            tfidf_dim=tfidf_dim,
            hidden_size=128,
            num_classes=3
        )

if __name__ == "__main__":
    # from llm_app import LLMApp

    # Example: initialize with BERT pipeline
    app = LLMApp(
        model_type="bert_ann",   # options: "tfidf_logreg", "tfidf_ann", "bert_ann", "fusion"
        model_name="distilbert-base-uncased",
        # device="cuda"            # use "cpu" if no GPU
    )
    app.main()
