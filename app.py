import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys


class ForensicDataMining:
    def __init__(self):
        self.data = None

    def load_data(self, file_path):
        """Load the dataset from a file (CSV)."""
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loaded successfully from {file_path}.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def describe_data(self):
        """Provide a summary of the dataset."""
        if self.data is not None:
            print("Data Summary:\n", self.data.describe())
            print("\nData Info:\n")
            print(self.data.info())
            print("\nMissing Values:\n", self.data.isnull().sum())
        else:
            print("No data loaded. Please load the data first.")

    def analyze_data(self):
        """Identify patterns and relationships in the data."""
        if self.data is not None:
            print("Analyzing data...\n")

            numeric_data = self.data.select_dtypes(include=[np.number])

            if numeric_data.empty:
                print("No numeric data available for correlation analysis.")
            else:
                print("Correlation Matrix:\n", numeric_data.corr())
        else:
            print("No data loaded. Please load the data first.")

    def visualize_data(self):
        """Provide various visualizations of the data."""
        if self.data is not None:
            print("Generating visualizations...\n")
            self.data.hist(figsize=(10, 8))
            plt.show()
        else:
            print("No data loaded. Please load the data first.")

    def classify_data(self):
        """Run a classification model (optional, depends on dataset)."""
        if self.data is not None:
            print("Running classification...\n")

            numeric_data = self.data.select_dtypes(include=[np.number])

            if numeric_data.empty:
                print("No numeric data available for classification.")
                return

            X = numeric_data.iloc[:, :-1]
            y = numeric_data.iloc[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            clf = RandomForestClassifier()

            clf.fit(X_train, y_train)

            accuracy = clf.score(X_test, y_test)
            print(f"Classification accuracy: {accuracy * 100:.2f}%")
        else:
            print("No data loaded. Please load the data first.")


def menu():
    app = ForensicDataMining()

    while True:
        print("\nForensic Data Mining Application")
        print("1. Load Forensic Data")
        print("2. Describe Data")
        print("3. Analyze Data")
        print("4. Classify Data")
        print("5. Visualize Data")
        print("6. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            file_path = input("Enter the path to the forensic data file: ")
            app.load_data(file_path)
        elif choice == "2":
            app.describe_data()
        elif choice == "3":
            app.analyze_data()
        elif choice == "4":
            app.classify_data()
        elif choice == "5":
            app.visualize_data()
        elif choice == "6":
            print("Exiting...")
            sys.exit()
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    menu()
