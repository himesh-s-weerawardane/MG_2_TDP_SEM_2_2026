# imports
import pandas as panda
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle

def nb_function(train_dataset, test_dataset, model_variant):
    # Read in training and test datasets
    train = panda.read_csv(train_dataset)
    test = panda.read_csv(test_dataset)

    # TF-IDF
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(train["text"])
    x_test = vectorizer.transform(test["text"])
    y_train = train["label"]
    y_test = test["label"]

    # Naive Bayes modelling
    # model = MultinomialNB()
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)

    # Fine tuning
    import numpy as np
    from sklearn.model_selection import GridSearchCV
    
    param_grid = {
        'alpha': np.logspace(-3, 1, 9),   # 0.001 to 10
        'fit_prior': [True, False],
    }

    clf = GridSearchCV(
        MultinomialNB(),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)

    model = clf.best_estimator_
    y_pred = model.predict(x_test)
    with open("fine_tuning.txt", "a") as f:
        f.write(f"Naive Bayes Data for model {model_variant}\n")
        f.write(f"best params: {clf.best_params_}\n")
        f.write(f"best score: {clf.best_score_}\n\n")

    # Save the model and vectorizer for frontend.
    with open(f"pkls/vectorizer/nb_vectorizer_{model_variant}.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(f"pkls/models/nb_{model_variant}.pkl", "wb") as f:
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
    plt.title(f"Confusion matrix for naive bayes model [{model_variant}]")
    plt.savefig(f"results/images/naive_bayes_confusion_matrix_model_{model_variant}.png")
    print(f"Confusion matrix plotted for naive bayes model [{model_variant}].")
    plt.close()

    # Write to file
    with open("results/scores.txt", "a") as f:
        f.write(f"Naive Bayes Data for model {model_variant}\n")
        f.write(f"Accuracy: {models_accuracy_score}\n")
        f.write(f"Precision: {models_precision_score}\n")
        f.write(f"Recall: {models_recall_score}\n")
        f.write(f"F1-score: {models_f1_score}\n")
        f.write(f"Confusion matrix data: {models_confusion_matrix.tolist()}\n\n")
    
    print(f"Naive bayes evaluation complete for Model {model_variant}.")

def nb_main(train_dataset, test_dataset, model_variant):
    nb_function(train_dataset, test_dataset, model_variant)