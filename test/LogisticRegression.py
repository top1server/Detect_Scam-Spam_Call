import sys
import os
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath('/Detect_scam-spam_call'))
from app.python.Models.LogisticRegressionClassifier import LogisticRegressionClassifier

if __name__ == "__main__":
    
    classifier = LogisticRegressionClassifier("D:\\Detect_Scam-Spam_Call\\app\\python\\Models\\dataset.json")
    texts, labels = classifier.load_data()
    texts = classifier.preprocess_texts(texts)
    lr_model, le, tfidf_vectorizer, report_lr, y_test, y_pred_lr = classifier.train(texts, labels)
    classifier.plot_metrics(report_lr)
    classifier.plot_confusion_matrix(y_test, y_pred_lr, le)
    predicted_label = classifier.predict_label("À, con nhớ làm bài tập sớm nhé, rồi đi ngủ đúng giờ.")
    print(predicted_label)
