# 🧠 Sentiment Analysis – AI & Machine Learning Project

This project leverages **Machine Learning (ML)** and **Deep Learning (DL)** techniques to classify sentiments in text data, focusing on **movie reviews** and **social media posts**.  
It evaluates multiple models – from **Naive Bayes** and **SVM** to **RNNs** and **CNNs** – and identifies the best-performing approach for accurate sentiment prediction.  

---

## 📌 Table of Contents
- [About the Project](#-about-the-project)
- [Objectives](#-objectives)
- [Methodology](#-methodology)
- [Tech Stack](#-tech-stack)
- [Installation](#️-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Conclusion](#-conclusion)
- [Future Work](#-future-work)
- [References](#-references)

---

## 🔍 About the Project
The **Sentiment Analysis Project** aims to classify textual data into **positive, negative, or neutral sentiments** using ML/DL algorithms.  

This project demonstrates:
- Robust **data preprocessing** (cleaning, normalization, tokenization, lemmatization).
- Evaluation of **classical ML models** (Naive Bayes, SVM).
- Implementation of **Deep Learning models** (RNN, CNN).
- Comparative **performance analysis** using metrics such as **accuracy, precision, recall, and F1-score**.

---

## 🎯 Objectives
1. Develop a robust ML/DL-based sentiment classifier.  
2. Preprocess and clean datasets for consistency and accuracy.  
3. Evaluate and compare multiple models to identify the most effective approach.  
4. Provide an intuitive and user-friendly way to visualize sentiment results.  
5. Offer clear documentation for reproducibility and learning.  

---

## 🛠 Methodology

### 📂 Data Collection
- **IMDB Movie Reviews Dataset** (50,000 labeled reviews).  
- Social media posts (optional extension).  

### 🔧 Data Preprocessing
- Remove noise (punctuation, special characters, stopwords).  
- Normalize case (lowercasing).  
- Tokenize sentences.  
- Apply **lemmatization/stemming**.  
- Convert text into **TF-IDF vectors** (for ML models).  
- Use **padded sequences** (for DL models).  

### 🤖 Model Selection
- **Naive Bayes** – Probabilistic classifier (baseline).  
- **Support Vector Machines (SVM)** – Finds optimal hyperplane for classification.  
- **Recurrent Neural Networks (RNN)** – Captures sequential dependencies.  
- **Convolutional Neural Networks (CNN)** – Captures local semantic patterns.  

### 📊 Training & Evaluation
- Dataset split into **training** and **testing** sets.  
- Feature extraction applied (TF-IDF, embeddings).  
- Evaluation metrics: **Accuracy, Precision, Recall, F1-score**.  
- **Cross-validation** to ensure generalizability.  

---

## 🖥 Tech Stack
- **Languages**: Python 3.x  
- **ML/DL Libraries**: Scikit-learn, TensorFlow, Keras  
- **NLP Tools**: NLTK  
- **Data Handling**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn  

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Haile-12/sentiment-analysis-ml-project.git
cd sentiment-analysis-ml-project

# Create a virtual environment
python -m venv venv

# Activate the environment
# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📈 Results
- **Naive Bayes** → Accuracy: `0.85`  
- **SVM** → Accuracy: `0.89` ✅ *Best ML Model*  
- **RNN** → Accuracy: `0.51` ⚠️ *Underperformed*  
- **CNN** → Accuracy: `0.87` ✅ *Best DL Model*  

---

## 📊 Visualizations
- **RNN Accuracy** → ~`0.50` (struggled to learn).  
- **CNN Accuracy** → ~`0.87` (consistent & reliable).  


---

## ✅ Conclusion
- The **CNN model** outperformed other models, making it the most suitable for sentiment classification tasks.  
- Proper **data preprocessing** and **model selection** proved essential for reliable results.  

---

## 🔮 Future Work
- 🔧 Improve **RNN performance** (hyperparameter tuning, GRU/LSTM variants).  
- 📊 Expand datasets with **diverse sources** (Twitter, product reviews).  
- 🌐 Develop an **interactive web dashboard** for real-time predictions.  
- 🤖 Explore **Transformer-based models** (BERT, RoBERTa, GPT).  

---

## 📚 References
- [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)  
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)  
- [Keras Documentation](https://keras.io/api/)  
