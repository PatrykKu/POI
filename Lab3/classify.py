import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def classify(csv_path):
    df = pd.read_csv(csv_path)

    X = df.drop('category', axis=1)
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = SVC(kernel='linear')
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Dokładność klasyfikacji: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    csv_path = "texture_features.csv"
    classify(csv_path)