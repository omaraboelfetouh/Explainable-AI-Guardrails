import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_prep_data():
    """
    Loads the Iris dataset from a reliable URL and prepares it for classification.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    dataset = pd.read_csv(url, names=column_names)
    
    # Map species to numerical values
    species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    dataset['species'] = dataset['species'].map(species_map)
    
    X = dataset.drop('species', axis=1)
    y = dataset['species']
    return X, y, species_map

def train_knn(X, y, n_neighbors=5):
    """
    Splits data and trains a KNN classifier.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    return knn, X_test, y_test

def evaluate_model(model, X_test, y_test, target_names):
    """
    Prints detailed evaluation metrics for the trained model.
    """
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
    # Pipeline execution
    X, y, species_map = load_and_prep_data()
    target_names = list(species_map.keys())
    
    knn_model, X_test, y_test = train_knn(X, y)
    evaluate_model(knn_model, X_test, y_test, target_names)