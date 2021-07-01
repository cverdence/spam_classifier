# Load libraries
import pandas as pd
import spam_classifier.utils as u
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Read data and explore
df = pd.read_table('data/SMSSpamCollection')
df.columns = ['label', 'text']
df.head()

df['label'] = df.label.map(u.label_converter)
print(df.shape)

# Training
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

# Naive Bayes
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

# Apply to test data
predictions = naive_bayes.predict(testing_data)

# Evaluate the model
print('Accuracy score: ', format(accuracy_score(predictions, y_test)))
print('Precision score: ', format(precision_score(predictions, y_test)))
print('Recall score: ', format(recall_score(predictions, y_test)))
print('F1 score: ', format(f1_score(predictions, y_test)))