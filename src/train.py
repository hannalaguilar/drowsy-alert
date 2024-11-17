import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


SELECTED_COLUMNS = ['jawOpen',
 'cheekPuff',
 'mouthLowerDownLeft',
 'jawRight',
 'mouthLowerDownRight',
 'mouthShrugLower',
 'mouthFunnel',
 'jawForward',
 'mouthClose',
 'mouthStretchLeft']


if __name__ == '__main__':

    files_and_labels = [
        ("Normal.csv", "normal"),
        # ("Talking.csv", "talking"),
        ("Yawning_cleaned2.csv", "yawning"),
    ]

    combined_csv = pd.concat(
        [pd.read_csv(file).assign(label=label) for file, label in files_and_labels],
        ignore_index=True
    )

    X = combined_csv[SELECTED_COLUMNS]
    y = combined_csv['label']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_valid)}")
    print(f"Test set size: {len(X_test)}")

    # Train a Random Forest model
    clf = RandomForestClassifier(random_state=42, max_depth=5)
    clf.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_valid_pred = clf.predict(X_valid)
    print("Validation set results:")
    print(classification_report(y_valid, y_valid_pred))

    # Evaluate the model on the test set
    y_test_pred = clf.predict(X_test)
    print("Test set results:")
    print(classification_report(y_test, y_test_pred))

    # Calculate accuracy on the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test set accuracy: {test_accuracy:.2f}")

    # save
    joblib.dump(clf, "random_forest_model.pkl")
