AI-Powered Fake News Detection

A machine learning and NLP project for classifying news articles as Real or Fake using text representations, engineered linguistic signals, and multiple supervised learning algorithms.

Note: The reported 100% test performance comes from a controlled synthetic dataset with highly separable features. It should not be interpreted as real-world fake-news detection accuracy.

📌 Project Overview

This project implements an end-to-end fake news classification pipeline, covering:

Manual text preprocessing and tokenization
Bag-of-Words and TF-IDF feature extraction
Precomputed 768-dimensional text embeddings
Five engineered numeric signals
Comparison of parametric, non-parametric, ensemble, and neural-network models
Stratified train/test evaluation
Feature importance and separability analysis

The project was completed as part of the Summer Internship Program in AI & ML, 2026 at the Indian Institute of Computing and Technology (IICT).

📊 Dataset

The dataset contains 1,200 news articles:

600 Real articles
600 Fake articles
14 columns
No missing values
Main Features
Feature	Description
title	News headline
full_text	Full article text
source_name	Publishing outlet
category	Article category
country	Country of origin
sentiment_score	Article tone/polarity
clickbait_score	Degree of sensationalized phrasing
emotional_word_ratio	Ratio of emotionally charged words
readability_score	Readability measure
source_trust_score	Publisher credibility score
text_embedding	Precomputed 768-dimensional embedding
label	0 = Real, 1 = Fake
🔧 Methodology
1. Text Preprocessing
Lowercasing
Removing URLs and HTML tags
Removing punctuation and digits
Whitespace normalization
Manual tokenization
Stopword removal
Removing single-character tokens
2. Feature Extraction

Three feature configurations were evaluated:

TF-IDF text features
Numeric features only
TF-IDF + numeric features

The numeric features were standardized using StandardScaler.

3. Machine Learning Models
K-Nearest Neighbors (KNN) — non-parametric
Logistic Regression — parametric
Random Forest — ensemble
Multi-Layer Perceptron (MLP) — neural network
4. Evaluation

The data was split using an 80/20 stratified train-test split with random_state=42.

Metrics used:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
📈 Results

All four models achieved 1.00 accuracy, precision, recall, and F1-score on the held-out test set across the evaluated feature configurations.

Model	TF-IDF	Numeric	Combined
KNN	1.00	1.00	1.00
Logistic Regression	1.00	1.00	1.00
Random Forest	1.00	1.00	1.00
MLP	1.00	1.00	1.00
⚠️ Important Interpretation

The perfect scores are primarily explained by strong structural separability in the dataset. Several engineered numeric features have non-overlapping ranges between the classes, and the source outlets are separated by label.

Therefore, the results demonstrate that the pipeline works correctly on the controlled dataset, rather than proving production-level fake-news detection performance.

🔍 Feature Importance

Random Forest analysis identified several useful signals, including:

emotional_word_ratio
source_trust_score
clickbait_score
Important text tokens such as detailed, official, analysis, believe, and revelation
🛠️ Tech Stack
Python
Pandas
NumPy
Scikit-learn
Natural Language Processing (NLP)
TF-IDF
Bag-of-Words
Machine Learning
Matplotlib
📁 Project Structure
fake-news-detection/
│
├── fake_news_detection-checkpoint.ipynb
├── Fake_News_Detection_Report_Final.pdf
├── README.md
│
└── data/
    └── dataset files
🚀 Getting Started
Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd fake-news-detection
Install dependencies
pip install pandas numpy scikit-learn matplotlib
Run the notebook

Open:

fake_news_detection-checkpoint.ipynb

using Jupyter Notebook, JupyterLab, or Google Colab.

🔮 Future Work
Test on real-world, noisier fake-news datasets
Cross-validation
Hyperparameter tuning
Probability calibration
Better generalization testing
Evaluation on unseen publishers and topics
Explore modern transformer-based NLP models
👨‍💻 Author

M. Sravan Kumar Varma
B.Tech — Computer Science and Engineering (AI & ML)
CMR Technical Campus, Hyderabad

📧 srvnkumr27@gmail.com

🔗 GitHub: https://github.com/srvnkumr27

📄 Project Report

AI-Powered Fake News Detection Using Text Classification: A Comparative Study of Parametric and Non-Parametric Machine Learning Models
