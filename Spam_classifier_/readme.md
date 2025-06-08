# 📩 SMS/Email Spam Classifier

A machine learning project that classifies messages as **spam** or **ham (not spam)** using natural language processing and an ensemble learning technique.

---

## 📌 Project Overview

This project utilizes a **Voting Classifier** combining multiple models to accurately predict whether a message is spam. It uses text preprocessing, TF-IDF vectorization, and an ensemble of classifiers to ensure robust performance.

---

## 💡 Features

- ✅ Clean and transform raw text data (tokenization, stemming, stopword removal)
- 📊 Convert text into numerical features using **TF-IDF Vectorization**
- 🤖 Predict using a **Voting Classifier (soft voting)** made up of:
  - Support Vector Machine (SVM)
  - Multinomial Naive Bayes (MNB)
  - Extra Trees Classifier (ETC)
- 🖥️ Web interface powered by **Streamlit**

---

## 🔧 Tech Stack

| Component          | Tool/Library             |
|--------------------|--------------------------|
| Language           | Python 3.12              |
| ML Libraries       | scikit-learn, NLTK       |
| Text Vectorization | TF-IDF (from sklearn)    |
| GUI/Web App        | Streamlit                |
| Model Persistence  | Pickle                   |

---

## 🧠 Model Description

```python
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(
    estimators=[
        ('svm', svc), 
        ('nb', mnb), 
        ('et', etc)
    ],
    voting='soft'
)



# 📊 SMS Spam Collection Dataset

This dataset is used for training and evaluating machine learning models that classify SMS messages as **spam** or **ham (not spam)**.

---

## 📁 Dataset Overview

The dataset contains labeled SMS messages with two main columns:

| Column | Description                  |
|--------|------------------------------|
| `v1`   | Label — `ham` or `spam`      |
| `v2`   | The actual message text      |

### Example Rows

| v1   | v2                                                                 |
|------|--------------------------------------------------------------------|
| ham  | Go until jurong point, crazy... Only in bugis n great world...    |
| spam | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May...  |
| ham  | Ok lar... Joking wif u oni...                                     |

---

## 🔍 Label Description

- **ham** – Legitimate, non-spam message.
- **spam** – Unwanted promotional or scam message.

---

## 🧹 Preprocessing Notes

Before training a model, the following preprocessing steps are recommended:

1. Convert text to lowercase.
2. Tokenize using NLTK.
3. Remove stopwords and punctuation.
4. Apply stemming (e.g., Porter Stemmer).
5. Vectorize using TF-IDF or CountVectorizer.

---

## 📦 File Format

- **File Type**: `.csv`
- **Encoding**: UTF-8
- **Columns**: 2 (Label, Message)

---




## 📈 Usage

- Spam detection research
- NLP preprocessing practice
- Text classification model training

---


