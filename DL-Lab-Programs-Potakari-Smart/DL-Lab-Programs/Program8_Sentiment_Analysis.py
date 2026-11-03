# --------------------------------------------------
# Author Name : POTAKARI SMART
# USN         : 1AM23CI107
# Branch      : CSE - AIML
# College     : AMC Engineering College, Bengaluru
# Subject     : Deep Learning Laboratory
# --------------------------------------------------

"""
Program 8 (Sentiment Analysis)
Read a dataset of text reviews and classify the reviews as
positive or negative.
"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Dataset of text reviews (1 = Positive, 0 = Negative)
positive_reviews = [
    "I loved this movie, it was fantastic",
    "The product is excellent and very useful",
    "Amazing experience, I really enjoyed it",
    "The service was wonderful and the staff was friendly",
    "Very happy with my purchase, it is great",
    "Great quality, I love this product",
    "The movie was amazing and the acting was brilliant",
    "Excellent service, I am very happy",
    "A wonderful film, I loved every minute",
    "Fantastic product, works perfectly and I love it",
    "The food was delicious and the service was excellent",
    "Best purchase ever, great quality and very happy",
    "I really enjoyed this film, it was amazing",
    "Wonderful experience, great staff and excellent food",
    "Love it, the product is fantastic and works great",
    "Superb movie, I am very happy I watched it",
    "Very good quality, I am happy with this purchase",
    "The service was great, I loved the food",
    "An excellent and enjoyable movie, I loved it",
    "Amazing product, I am very happy and love the quality",
]
negative_reviews = [
    "This product is terrible",
    "I hated this movie, it was awful",
    "Very disappointing experience, bad service",
    "The product broke after one day, very poor quality",
    "Poor quality and terrible service",
    "Awful movie, I hated every minute",
    "The product was awful and I am very disappointed",
    "Worst purchase ever, poor quality and very unhappy",
    "Terrible service, the staff was rude and bad",
    "I am disappointed, the product is bad and stopped working",
    "A boring and terrible film, I hated it",
    "Bad quality, I hate this product",
    "The food was horrible and the service was terrible",
    "Awful experience, rude staff and poor food",
    "Hate it, the product is terrible and broke quickly",
    "Worst movie, I am very unhappy I watched it",
    "Very bad quality, I am disappointed with this purchase",
    "The service was bad, I hated the food",
    "A boring and disappointing movie, I hated it",
    "Terrible product, I am very unhappy and hate the quality",
]

reviews = positive_reviews + negative_reviews
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    reviews,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels,
)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate on test data
y_pred = model.predict(X_test_tfidf)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Test new reviews
new_reviews = [
    "The movie was amazing and I loved it",
    "The product was awful and disappointing",
    "I am very happy with this purchase",
]
new_reviews_tfidf = vectorizer.transform(new_reviews)
predictions = model.predict(new_reviews_tfidf)

# Display results
for review, prediction in zip(new_reviews, predictions):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("\nReview:", review)
    print("Sentiment:", sentiment)
