# NLP Model Evaluation

## 1. Intent Classification

### Dataset

The intent classification dataset contains **400 sentences** divided equally into four classes:

* Question: 100 sentences
* Command: 100 sentences
* Complaint: 100 sentences
* Greeting: 100 sentences

The dataset was split into:

* Training samples: 300
* Testing samples: 100

### Accuracy Comparison

| Model                                     | Accuracy |
| ----------------------------------------- | -------- |
| Naive Bayes                               | 100%     |
| Logistic Regression (Before GridSearchCV) | 100%     |
| Logistic Regression (After GridSearchCV)  | 100%     |

### Best Parameters

* C = 0.01
* Solver = lbfgs
* Class Weight = None
* TF-IDF ngram_range = (1,1)

### Best Cross-Validation Accuracy

**100%**

### Best Model

All three models achieved the same test accuracy of 100%. Naive Bayes was selected as the final model among the tied models.

---

## 2. Sentiment Analysis

### Dataset

The sentiment analysis dataset contains **300 sentences** divided equally into three classes:

* Positive: 100 sentences
* Negative: 100 sentences
* Neutral: 100 sentences

The dataset was split into:

* Training samples: 225
* Testing samples: 75

### Accuracy Comparison

| Model                                     | Accuracy |
| ----------------------------------------- | -------- |
| Logistic Regression (Before GridSearchCV) | 100%     |
| Logistic Regression (After GridSearchCV)  | 100%     |

### Baseline Cross-Validation Accuracy

**100%**

### Best Parameters

* C = 1
* Solver = lbfgs
* Class Weight = None
* TF-IDF ngram_range = (1,1)
* TF-IDF min_df = 1

### Best GridSearchCV Cross-Validation Accuracy

**100%**

### Best Model

The Baseline Logistic Regression model was selected because both the baseline and optimized models achieved the same test accuracy of 100%.

---

## Conclusion

* Expanded the intent classification dataset to 400 sentences.
* Expanded the sentiment analysis dataset to 300 sentences.
* Divided the datasets into balanced training and testing sets.
* Used TF-IDF for text feature extraction.
* Applied Naive Bayes and Logistic Regression for intent classification.
* Applied Logistic Regression for sentiment analysis.
* Used GridSearchCV to identify the best model parameters.
* The intent classification models achieved 100% accuracy on the current test dataset.
* The sentiment analysis models achieved 100% accuracy on the current test dataset.
* Both tasks achieved a best cross-validation accuracy of 100%.
* Naive Bayes was selected as the final intent classification model.
* Baseline Logistic Regression was selected as the final sentiment analysis model
