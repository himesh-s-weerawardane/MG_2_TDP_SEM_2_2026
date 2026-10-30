# imports
import pandas as panda
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle

def svm_function(train_dataset, test_dataset, model_variant):
    # Read in training and test datasets
    train = panda.read_csv(train_dataset)
    test = panda.read_csv(test_dataset)

    # TF-IDF
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(train["text"])
    x_test = vectorizer.transform(test["text"])
    y_train = train["label"]
    y_test = test["label"]

    # SVM modelling
    model = svm.LinearSVC(C=1.0, loss='hinge', max_iter=15000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Save the model and vectorizer for frontend.
    with open(f"pkls/vectorizer/svm_vectorizer_{model_variant}.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(f"pkls/models/svm_{model_variant}.pkl", "wb") as f:
        pickle.dump(model, f)

    # Evaluate the prediction
    models_accuracy_score = accuracy_score(y_test, y_pred)
    models_precision_score = precision_score(y_test, y_pred)
    models_recall_score = recall_score(y_test, y_pred)
    models_f1_score = f1_score(y_test, y_pred)
    models_confusion_matrix = confusion_matrix(y_test, y_pred)

    # Print to terminal
    print("Accuracy: ", models_accuracy_score)
    print("Precision: ", models_precision_score)
    print("Recall: ", models_recall_score)
    print("F1-score: ", models_f1_score)
    print("Confusion matrix: ", models_confusion_matrix.tolist())

    # Plot the matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion matrix for SVM model [{model_variant}]")
    plt.savefig(f"results/images/SVM_confusion_matrix_model_{model_variant}.png")
    print(f"Confusion matrix plotted for SVM model [{model_variant}].")
    plt.close()

    # Write to file
    with open("results/scores.txt", "a") as f:
        f.write(f"SVM Data for model {model_variant}\n")
        f.write(f"Accuracy: {models_accuracy_score}\n")
        f.write(f"Precision: {models_precision_score}\n")
        f.write(f"Recall: {models_recall_score}\n")
        f.write(f"F1-score: {models_f1_score}\n")
        f.write(f"Confusion matrix data: {models_confusion_matrix.tolist()}\n\n")
    
    print(f"SVM evaluation complete for Model {model_variant}.")

def svm_main(train_dataset, test_dataset, model_variant):
    svm_function(train_dataset, test_dataset, model_variant)