# Sentiment Classification Project - Movie Review Analysis

## 1. General Description

This project is a work demonstrating the ability to solve a Data Science (DS) task and apply Machine Learning Engineering (MLE) skills. 
The project involves building a binary sentiment classification model to classify movie reviews as either positive or negative. 
The solution includes both a Data Science (DS) part, focused on model development, and an MLE part, focused on creating a production-ready, 
Dockerized solution.

## 2. DS Part: Binary Sentiment Classification

### 2.1. Task

*   **Task:** Binary sentiment classification
*   **Target Variable:** Sentiment (positive/negative)
*   **Dataset:** 50,000 polar movie reviews (train.csv and test.csv)
*   **Metric:** Accuracy (target >= 0.85)

### 2.2. Data

The dataset consists of movie reviews and their corresponding sentiment labels (positive or negative). The data is provided in 
two CSV files:

*   `train.csv`: Used for training the sentiment classification model (40,000 reviews).
*   `test.csv`: Used for evaluating the model's performance (10,000 reviews).

The datasets should be downloaded from the provided URLs using the `data_loader.py` script.

### 2.3. EDA and Conclusions

*   **Data Overview:** The datasets contain two columns: 'review' (text of the movie review) and 'sentiment' (sentiment label: 
'positive' or 'negative'). Both columns are of object type (strings).
*   **Sentiment Distribution:** The sentiment labels in both the training and test datasets are perfectly balanced, 
with 50% positive and 50% negative reviews. This balanced class distribution is ideal for training and evaluation, 
and accuracy is a suitable metric.
*   **Missing Values:**  Exploratory Data Analysis revealed that there are **no missing values** in either the training or test datasets,
simplifying the data preprocessing pipeline.
*   **Review Text Length:** Basic descriptive statistics were examined for the 'review' column, providing initial insights into 
the text data.

**Conclusion from EDA:** The datasets are clean, well-structured, and balanced, providing a solid foundation for building a sentiment 
classification model. The absence of missing values is particularly beneficial. The balanced sentiment distribution means we can 
confidently use accuracy as our primary evaluation metric. We can now proceed to text preprocessing and feature engineering.

### 2.4. Text Preprocessing and Feature Engineering

*   **Tokenization:** Word tokenization was performed using NLTK's `word_tokenize`. This breaks down each movie review into individual 
words, which is essential for subsequent preprocessing steps. Lowercasing was applied to ensure consistency.
*   **Stop-word Removal:** Stop words (common English words like "the", "is", "and") and punctuation were removed using 
NLTK's `stopwords` corpus and `string.punctuation`. This step reduces noise and dimensionality in the text data.
*   **Stemming vs. Lemmatization:** Both stemming (using Porter Stemmer) and lemmatization (using WordNet Lemmatizer) were compared. 
While both techniques reduce words to their root form, **lemmatization was chosen** as it generally provides more linguistically 
accurate base forms compared to stemming, which can sometimes result in non-words.
*   **Vectorization:** Two vectorization techniques were compared: CountVectorizer (Bag of Words) and TfidfVectorizer (TF-IDF). 
**TfidfVectorizer was selected** as it performed slightly better in initial model experiments. TF-IDF weights words based on their 
frequency in a document and across the corpus, emphasizing words that are more important to specific reviews, which is beneficial 
for sentiment classification.

**Conclusion from Feature Engineering:** Text preprocessing focused on cleaning and normalizing the review text through tokenization, 
stop-word removal, and lemmatization. TfidfVectorizer was chosen for feature extraction due to its ability to weight words by importance,
which is expected to improve model performance for sentiment classification.

### 2.5. Modeling

*   **Baseline Model: Logistic Regression:** Logistic Regression was used as a baseline model due to its simplicity and effectiveness 
in binary classification tasks. It achieved an accuracy of approximately **0.8972** on the test set when using TfidfVectorizer and 
Lemmatization.
*   **Explored Models:** In addition to Logistic Regression, two other models were explored:
    *   **Support Vector Machine (SVM) (LinearSVC):** SVM model outperformed Logistic Regression, achieving higher accuracy.
    *   **Naive Bayes (MultinomialNB):** Multinomial Naive Bayes, suitable for count-based features, also provided competitive results.
*   **Model Comparison:** A comparison of different models and vectorization techniques revealed the following summarized accuracies:

    | Model                  | Vectorizer      | Preprocessing | Accuracy |
    |------------------------|-----------------|---------------|----------|
    | Logistic Regression    | CountVectorizer | Stemmed       | 0.8871   |
    | Logistic Regression    | TfidfVectorizer | Stemmed       | 0.8970   |
    | Logistic Regression    | CountVectorizer | Lemmatized    | 0.8913   |
    | Logistic Regression    | TfidfVectorizer | Lemmatized    | 0.8972   |
    | SVM                    | CountVectorizer | Stemmed       | 0.8677   |
    | SVM                    | TfidfVectorizer | Stemmed       | 0.8973   |
    | SVM                    | CountVectorizer | Lemmatized    | 0.8747   |
    | SVM                    | TfidfVectorizer | Lemmatized    | **0.9013**   |
    | MultinomialNB          | CountVectorizer | Stemmed       | 0.8550   |
    | MultinomialNB          | CountVectorizer | Lemmatized    | 0.8582   |
    | MultinomialNB          | TfidfVectorizer | Stemmed       | 0.8624   |
    | MultinomialNB          | TfidfVectorizer | Lemmatized    | 0.8692   |

*   **Best Model Selection:** Based on the comparison, **SVM with TfidfVectorizer on lemmatized text was chosen as the best model** 
due to achieving the highest accuracy of **0.9013**, comfortably exceeding the target accuracy of 0.85. SVMs are well-suited for
high-dimensional text data and, combined with TF-IDF vectorization and lemmatization, proved to be the most effective configuration 
for this sentiment classification task.

**Overall Performance Evaluation:**

*   **Best Model:** SVM with TfidfVectorizer (Lemmatized Text)
*   **Accuracy on Test Set:** **0.9013**

    *Achieved accuracy is greater than the required 0.85.*

### 2.6. Potential Business Applications and Value for Business

*   **Customer Feedback Analysis:** Sentiment analysis can be used to automatically analyze customer feedback from movie reviews, 
surveys, and social media to understand customer opinions about films. This can help studios and filmmakers gauge audience reception 
and identify areas for improvement.
*   **Social Media Monitoring:** The model can monitor social media platforms for real-time sentiment towards movies, actors, or studios. 
This allows for quick identification of public perception shifts and proactive reputation management.
*   **Market Research:** Sentiment analysis can be applied to analyze competitor movie reviews, helping studios understand audience 
preferences and market trends. This can inform decisions about movie genres, casting, and marketing strategies.
*   **Improved Decision Making:** By providing quantifiable sentiment insights, the model can support data-driven decision-making 
in the entertainment industry, from content creation to marketing and distribution strategies.

**Value for Business:** Implementing sentiment analysis for movie reviews provides significant business value by automating 
the process of understanding audience opinions at scale. This leads to cost and time savings compared to manual analysis, enables 
data-driven decision-making, and enhances competitiveness by providing timely and actionable insights into audience preferences 
and market sentiment. This can ultimately contribute to creating more successful and audience-aligned movie productions and marketing 
campaigns.

## 3. MLE Part: Productionizing the Solution with Docker

### 3.1. Repository Structure

```
/data-science-task/
|--data/ # Data directory (ignored by Git, data should be downloaded by data_loader.py)
|     |--raw/
|--notebooks/ # Notebook directory containing the Jupyter notebook
|--src/
|     |--train/
|     |     |--train.py
|     |     |--Dockerfile
|     |--inference/
|     |     |--run_inference.py
|     |     |--Dockerfile
|     |--data_loader.py
|--outputs/ # Output directory (ignored by Git, for models, predictions, metrics)
|     |--models/
|     |     |--model.pkl
|     |--predictions/
|     |     |--predictions.csv
|     |     |--metrics.txt
|--README.MD
|--requirements.txt
```

### 3.2. How to Run the Solution with Docker (PowerShell)

**Comprehensive, step-by-step instructions to run solution using Docker in PowerShell**

**Prerequisites:**

*   Docker Desktop installed and running on system.
*   PowerShell terminal.

**Steps:**

1.  **Clone the Repository:**

    Open Visual Studio Code and follow these steps:
       
    1. Open VSCode and select File > Open Folder. Choose the folder where you want to store the project.
    2. Once the folder is open, launch the built-in terminal by selecting Terminal > New Terminal. The terminal will open 
    in your project directory using PowerShell.
    3. In the integrated PowerShell terminal, run the following commands:

    ```powershell
    git clone https://github.com/MaksymTkachenko1/data-science-task.git
    cd data-science-task
    ```

2.  **Build the Training Image:**

    In **PowerShell**, run the following command from the project root directory (`data-science-task`):

    ```powershell
    docker build -f ./src/train/Dockerfile -t sentiment-train .
    ```

3.  **Run `data_loader.py` to Download Data:**

    ```powershell
    docker run -v ${PWD}/data:/app/data sentiment-train python /app/src/data_loader.py
    ```
    *This command downloads and extracts the datasets into the `data/raw` directory in project.*

4.  **Run Training:**

    ```powershell
    docker run -v ${PWD}/data:/app/data -v ${PWD}/outputs:/app/outputs sentiment-train
    ```
    *This command trains the sentiment classification model, saves the trained model to `outputs/models/model.pkl`, and saves 
    evaluation metrics to `outputs/predictions/metrics.txt`.*

5.  **Build the Inference Image:**

    ```powershell
    docker build -f ./src/inference/Dockerfile -t sentiment-inference .
    ```

6.  **Run Inference:**

    ```powershell
    docker run -v ${PWD}/data:/app/data -v ${PWD}/outputs:/app/outputs sentiment-inference
    ```
    *This command runs inference on the test dataset using the trained model. Predictions are saved to 
    `outputs/predictions/predictions.csv`.*

### 3.3. Quickstart Instructions (PowerShell)

**A very brief, quickstart guide for experienced users using PowerShell**

For users familiar with Docker and PowerShell, here's a quick guide how to run the solution:

1.  Choose the folder where you want to clone the repository.
2.  Clone the repository: `git clone https://github.com/MaksymTkachenko1/data-science-task.git` 
3.  Navigate to the project root directory: `cd data-science-task` 
4.  Build the training image: `docker build -f ./src/train/Dockerfile -t sentiment-train .`
5.  Run data loader: `docker run -v ${PWD}/data:/app/data sentiment-train python /app/src/data_loader.py`
6.  Run training: `docker run -v ${PWD}/data:/app/data -v ${PWD}/outputs:/app/outputs sentiment-train`
7.  Build the inference image: `docker build -f ./src/inference/Dockerfile -t sentiment-inference .`
8.  Run inference: `docker run -v ${PWD}/data:/app/data -v ${PWD}/outputs:/app/outputs sentiment-inference`

Check `outputs/predictions/predictions.csv` for predictions and `outputs/predictions/metrics.txt` for accuracy. Hooray!

## Created by

Maksym Tkachenko
