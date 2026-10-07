# imports
import pandas as panda
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def lr_function(train_dataset, test_dataset, model_variant):
    # Extracting data
    train = panda.read_csv(train_dataset)
    test = panda.read_csv(test_dataset)

    # TF-IDF
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(train["text"])
    x_test = vectorizer.transform(test["text"])
    y_train = train["label"]
    y_test = test["label"]

    # Logistic regression modelling
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Evaluate the model
    models_accuracy_score = accuracy_score(y_test, y_pred)
    models_precision_score = precision_score(y_test, y_pred)
    models_recall_score = recall_score(y_test, y_pred)
    models_f1_score = f1_score(y_test, y_pred)
    models_confusion_matrix = confusion_matrix(y_test, y_pred)

    # Print scores to terminal
    print("Accuracy: ", models_accuracy_score)
    print("Precision: ", models_precision_score)
    print("Recall: ", models_recall_score)
    print("F1-score: ", models_f1_score)
    print("Confusion matrix: ", models_confusion_matrix.tolist())

    # Plot the matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion matrix for logsitic regression model [{model_variant}]")
    plt.savefig(f"results/images/logsitic_regression_confusion_matrix_model_{model_variant}.png")
    print(f"Confusion matrix plotted for logistic regression model [{model_variant}].")
    plt.close()

    # Write to file
    with open("results/scores.txt", "a") as f:
        f.write(f"Logisitic Regression Data for model {model_variant}\n")
        f.write(f"Accuracy: {models_accuracy_score}\n")
        f.write(f"Precision: {models_precision_score}\n")
        f.write(f"Recall: {models_recall_score}\n")
        f.write(f"F1-score: {models_f1_score}\n")
        f.write(f"Confusion matrix data: {models_confusion_matrix.tolist()}\n\n")
    
    print(f"Logistic regression evaluation complete for Model {model_variant}.")

def lr_main(train_dataset, test_dataset, model_variant):
    lr_function(train_dataset, test_dataset, model_variant)