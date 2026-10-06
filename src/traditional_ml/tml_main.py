# imports
from logistic_regression import lr_main
from naive_bayes import nb_main

# Dictionaries for models.
model_A = {'model': "A", 'train': "data/processed/tweet_eval_train.csv", 'test': "data/processed/stock_emotions_test.csv"}
model_B = {'model': "B", 'train': "data/processed/stock_emotions_train.csv", 'test': "data/processed/stock_emotions_test.csv"}
model_C = {'model': "C", 'train': "data/processed/mixed_train.csv", 'test': "data/processed/stock_emotions_test.csv"}

# List of all models to be looped through for traditional ml.
all_models = [model_A, model_B, model_C]

# Function that calls the models with the relevant dataset sent in as a param.
for indiv_model in all_models:
    lr_main(train_dataset=indiv_model['train'], test_dataset=indiv_model['test'], model_variant=indiv_model['model'])
    nb_main(train_dataset=indiv_model['train'], test_dataset=indiv_model['test'], model_variant=indiv_model['model'])

# Final print
print("All collected data and images stored inside the results directory.")
