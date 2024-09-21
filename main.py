import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('ODI_match_info.csv')

# We only care about team1, team2, venue as input and winner as the output
df = df[['team1', 'team2', 'venue', 'winner']]

# Handle missing values by dropping rows where 'winner' is NaN
df = df.dropna(subset=['winner'])

# Define the features (X) and the target (y)
X = df[['team1', 'team2', 'venue']]
y = df['winner']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a preprocessing and modeling pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['team1', 'team2', 'venue'])
    ])

# Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Build the pipeline with preprocessing and model training
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Train the model
pipeline.fit(X_train, y_train)

# Save the model to a file
joblib.dump(pipeline, 'odi_winner_prediction_model.pkl')
print("Model saved as 'odi_winner_prediction_model.pkl'")

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Example Prediction Function using the trained model
def predict_winner(team1, team2, venue):
    prediction = pipeline.predict(pd.DataFrame([[team1, team2, venue]], columns=['team1', 'team2', 'venue']))
    return prediction[0]

# Test the function with some inputs
team1_input = 'Pakistan'
team2_input = 'Australia'
venue_input = 'Melbourne Cricket Ground'

predicted_winner = predict_winner(team1_input, team2_input, venue_input)
print(f'Predicted Winner for {team1_input} vs {team2_input} at {venue_input}: {predicted_winner}')

# Load the saved model from the file
loaded_model = joblib.load('odi_winner_prediction_model.pkl')
print("Model loaded from 'odi_winner_prediction_model.pkl'")

# Make a prediction using the loaded model
def predict_winner_from_loaded_model(team1, team2, venue):
    prediction = loaded_model.predict(pd.DataFrame([[team1, team2, venue]], columns=['team1', 'team2', 'venue']))
    return prediction[0]

# Test the function with loaded model
team1_input = 'India'
team2_input = 'Australia'
venue_input = 'Melbourne Cricket Ground'

predicted_winner_loaded_model = predict_winner_from_loaded_model(team1_input, team2_input, venue_input)
print(f'Predicted Winner (Loaded Model) for {team1_input} vs {team2_input} at {venue_input}: {predicted_winner_loaded_model}')
