# FAKE NEWS DETECTION

This repository contains the implementation of a Fake News Detection System using various machine learning and natural language processing techniques. The project leverages machine learning models like LSTM, SVM, KNN, Naive Bayes, Random Forest, and Decision Trees, alongside feature extraction and visualization tools to identify and classify news articles whether they are genuine or fake.

## Introduction

The proliferation of fake news poses a significant threat to public discourse and democratic processes. This project addresses this critical concern by developing a robust machine learning model for accurate fake news detection. Our approach integrates natural language processing and machine learning to identify misinformation effectively.

## Concepts Used
The project employs the following models and techniques:

- LSTM (Long Short-Term Memory): Effective for analyzing sequences of text data.
- SVM (Support Vector Machine): Useful for text classification, handling high-dimensional data.
- KNN (K-Nearest Neighbors): Classifies data based on the majority class of its k-nearest neighbors.
- Naive Bayes: Probabilistic algorithm efficient for text classification tasks.
- Random Forest: Ensemble learning method that constructs multiple decision trees.
- Decision Tree: Tree-like model for classification tasks.
- Feature Extraction: Techniques like TF-IDF and word embeddings for transforming text data.

**Data Analysis**

Exploratory data analysis (EDA) was performed to understand the distribution of true and fake news based on categories and dates. This involved visualizing the count of articles over time and across different subjects.

**Data Cleaning**

Data cleaning steps included merging titles and text, removing stopwords, lemmatization, and oversampling to address class imbalance. The text data underwent tokenization and padding before model training.

**Model Training Results**

The LSTM model was developed using TensorFlow and Keras, showing high accuracy in detecting fake news. Additional models like SVM, Decision Tree, KNN, and Random Forest were trained and evaluated using accuracy metrics and confusion matrices.

**Feature Extraction**

Techniques like N-grams, word embeddings, cosine similarity, and readability scores were used to enhance feature representation, measure textual similarity, and evaluate linguistic complexity.

**Visualization**

Visualization techniques such as precision-recall curves and confusion matrices were used to assess model performance. These visualizations provided insights into the strengths and weaknesses of the models in distinguishing genuine from fake news articles.

**Logistic Regression**

Logistic Regression was employed, and its performance was evaluated using confusion matrices and other visualization techniques.

**Transformers**

BERT (Bidirectional Encoder Representations from Transformers) was fine-tuned for the fake news detection task. BERT's ability to understand context and relationships between words significantly improved prediction accuracy.

## Packages

```
pip install sweetviz
pip install wordcloud
pip install textstat
```

## Installation

**Clone the repository:**

```
git clone https://github.com/prani954/FAKE_NEWS_DETECTION.git
cd FAKE_NEWS_DETECTION
```

## Usage

1. Preprocess the data:

   - Clean and prepare the dataset using the provided scripts.

2. Train the models:

   - Run the training scripts to train LSTM, SVM, KNN, Naive Bayes, Random Forest, and Decision Tree models.

3. Evaluate the models:

   - Use the evaluation scripts to generate accuracy metrics, confusion matrices, and other visualizations.

4. Visualize results:

   - Generate and analyze visualizations like word clouds, correlation matrices, and performance graphs.

## Contact

For any queries, please contact - vbhumi722@gmail.com
