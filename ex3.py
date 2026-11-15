
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('emails.csv')

print("Dataset shape:", df.shape)
# print(df.head())

# Separate features and target
X = df.drop(['Email No.', 'Prediction'], axis=1)
y = df['Prediction']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create and train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict test data
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
