# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:16:58 2023

@author: Francis Agbana
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample training data - you would replace this with your dataset
documents = [ 
    "This is the original document.",
    "This is a very similar document.",
    "This is a completely different document."
]

# Labels (0 for original, 1 for potentially plagiarized)
labels = [0, 1, 0]

# Preprocess the text (you can enhance this preprocessing)
def preprocess(text):
    text = text.lower()
    return text

# Create a Bag of Words (BoW) vectorizer
vectorizer = CountVectorizer(preprocessor=preprocess)

# Transform the documents into BoW features
X = vectorizer.fit_transform(documents)

# Initialize a Naive Bayes classifier
classifier = MultinomialNB()

# Fit the classifier with the BoW features and labels
classifier.fit(X, labels)

# Sample text to check for plagiarism
text_to_check = "This is a very similar document."

# Preprocess the text for checking
text_to_check = preprocess(text_to_check)

# Transform the text to BoW features
text_vector = vectorizer.transform([text_to_check])

# Predict the label for the text
predicted_label = classifier.predict(text_vector)

# Output the result
if predicted_label[0] == 0:
    print("The text is not plagiarized.")
else:
    print("The text is potentially plagiarized.")

