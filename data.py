import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.cluster import KMeans
import numpy as np

# Load BERT model and tokenizer for sequence classification (fine-tuned on email classification)
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example data: Email dataset
emails = [
    "Please update my account details.",
    "I have a question about my recent transaction.",
    "Can you help me reset my password?"
]

# Simulated transaction data
transaction_data = {
    'customer_id': [1, 2, 3, 4, 5],
    'transaction_amount': [100.0, 250.0, 50.0, 300.0, 150.0],
    'fraud_status': ['No', 'No', 'Yes', 'No', 'No']
}
df_transactions = pd.DataFrame(transaction_data)

# Function for email classification using BERT
def classify_email(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predicted_class = "Other"
    if outputs.logits.argmax().item() == 2:
        predicted_class = 'Account Update'
    elif outputs.logits.argmax().item() == 1:
        predicted_class = 'Transaction Inquiry'
    elif outputs.logits.argmax().item() == 0:
        predicted_class = 'Password Reset'
    return predicted_class

# Classify emails
classified_emails = [(email, classify_email(email)) for email in emails]

# Function for generating responses
def generate_response(category):
    if category == 'Account Update':
        return "Sure, please provide your account details and I'll update them for you."
    elif category == 'Transaction Inquiry':
        return "I can assist you with your recent transaction. Please provide more details."
    elif category == 'Password Reset':
        return "To reset your password, please follow the instructions on our website."
    else:
        return "Thank you for contacting us. We will get back to you shortly."

# Generate responses for classified emails
responses = [(email, generate_response(category)) for email, category in classified_emails]

# Function for providing transaction insights
def provide_transaction_insights():
    # Analyze transaction data
    avg_transaction_amount = df_transactions['transaction_amount'].mean()
    total_fraud_cases = df_transactions[df_transactions['fraud_status'] == 'Yes'].shape[0]

    # Cluster transaction amounts using KMeans
    X = df_transactions[['transaction_amount']].values
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    cluster_centers = kmeans.cluster_centers_

    insights = {
        'average_transaction_amount': avg_transaction_amount,
        'total_fraud_cases': total_fraud_cases,
        'transaction_clusters': cluster_centers.tolist()
    }

    return insights

# Provide transaction insights
transaction_insights = provide_transaction_insights()

# Display results
df_responses = pd.DataFrame(responses, columns=['Email', 'Response'])
print("Automated Responses:")
print(df_responses)
print("\nTransaction Insights:")
print(transaction_insights)
