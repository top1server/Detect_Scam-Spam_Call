import json
import re
import os
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

class LogisticRegressionClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.label_map = {'ham': 0, 'spam': 1}
        self.stop_words = set([
            "bị", "bởi", "cả", "các", "cái", "cần", "càng", "chỉ", "chiếc", "cho", "chứ", "chưa", "chuyện",
            "có", "có_thể", "cứ", "của", "cùng", "cũng", "đã", "đang", "đây", "để", "đến_nỗi", "đều", "điều",
            "do", "đó", "được", "dưới", "gì", "khi", "không", "là", "lại", "lên", "lúc", "mà", "mỗi", "này",
            "nên", "nếu", "ngay", "nhiều", "như", "nhưng", "những", "nơi", "nữa", "phải", "qua", "ra", "rằng",
            "rằng", "rất", "rồi", "sau", "sẽ", "so", "sự", "tại", "theo", "thì", "trên", "trước", "từ", "từng",
            "và", "vẫn", "vào", "vậy", "vì", "việc", "với", "vừa", "chà"
        ])
        self.lr_model = None
        self.le = None
        self.tfidf_vectorizer = None
    
    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Tệp dữ liệu không tồn tại: {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        texts, labels = [], []
        for item in data:
            parts = item.split(',', 1)
            if len(parts) == 2:
                label_str, text = parts
                label = self.label_map.get(label_str.strip().lower())
                if label is not None:
                    labels.append(label)
                    texts.append(text.strip().lower())
        return texts, labels

    def preprocess_text(self, text):
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\d+', 'num', text)
        tokens = [token for token in text.split() if token not in self.stop_words]
        return ' '.join(tokens)

    def preprocess_texts(self, texts):
        return [self.preprocess_text(text) for text in tqdm(texts, desc="Processing texts")]

    def train(self, texts, labels):
        # Chia dữ liệu thành tập huấn luyện và kiểm tra
        texts_train, texts_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.25, stratify=labels, random_state=42
        )

        # Vector hóa và cân bằng dữ liệu
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(texts_train)

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

        self.le = LabelEncoder()
        y_train_enc = self.le.fit_transform(y_train_resampled)
        y_test_enc = self.le.transform(y_test)

        X_train_resampled, y_train_enc = shuffle(X_train_resampled, y_train_enc, random_state=42)

        # Huấn luyện mô hình Logistic Regression
        self.lr_model = LogisticRegression(C=1.0, max_iter=1000)
        self.lr_model.fit(X_train_resampled, y_train_enc)

        # Dự đoán và đánh giá mô hình
        X_test_tfidf = self.tfidf_vectorizer.transform(texts_test)
        y_pred_enc_lr = self.lr_model.predict(X_test_tfidf)
        y_pred_lr = self.le.inverse_transform(y_pred_enc_lr)

        report_lr = classification_report(y_test, y_pred_lr, output_dict=True, zero_division=0)

        return self.lr_model, self.le, self.tfidf_vectorizer, report_lr, y_test, y_pred_lr
    
    def predict_label(self, text):
        if not self.lr_model or not self.le or not self.tfidf_vectorizer:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi phương thức 'train' trước.")

        # Tiền xử lý văn bản
        processed_text = self.preprocess_text(text)
    
        # Chuyển đổi văn bản thành đặc trưng với TF-IDF
        text_tfidf = self.tfidf_vectorizer.transform([processed_text])
        
        # Dự đoán nhãn với mô hình
        prediction = self.lr_model.predict(text_tfidf)
        
        # Chuyển đổi nhãn dự đoán thành tên nhãn nếu cần
        predicted_label = self.le.inverse_transform(prediction)
        
        return predicted_label[0]
    
    def plot_metrics(self, report_lr):
        categories = list(report_lr.keys())[:-3]
        plt.figure(figsize=(8, 6))
        plt.plot(categories, [report_lr[cat]['precision'] for cat in categories], marker='o', label='Precision', color='blue')
        plt.plot(categories, [report_lr[cat]['recall'] for cat in categories], marker='o', label='Recall', color='green')
        plt.plot(categories, [report_lr[cat]['f1-score'] for cat in categories], marker='o', label='F1-Score', color='red')
        plt.ylim(0, 1)
        plt.title('Logistic Regression Metrics')
        plt.ylabel('Score')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_test, y_pred_lr, le):
        cm = confusion_matrix(y_test, y_pred_lr, labels=le.classes_)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
        plt.title('Logistic Regression Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()
