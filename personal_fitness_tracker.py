import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('fitness.csv')

# Convert gender to numeric
df['Gender_male'] = df['Gender'].apply(lambda x: 1 if x == 'male' else 0)

# Features and target
X = df[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Gender_male']]
y = df['Calories']

# Train/test split (optional for testing purposes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('personal_fitness_tracker.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as 'personal_fitness_tracker.pkl'")
