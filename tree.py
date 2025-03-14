# tree.py
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # progress bar library

def get_features(df):
    """
    Extracts 124 technical indicators from the DataFrame.
    It is assumed that these indicators are located in columns 13 to 136 (i.e., columns 12 to 135 in 0-indexing),
    and converts them into a NumPy array of type float64.
    """
    return df.iloc[:, 12:136].values.astype(np.float64)

def train_ensemble(train_df):
    """
    Accepts the training data DataFrame, extracts the 124 factor data (assumed to be in columns 13 to 136) 
    and the labels, constructs 100 decision trees, and returns a list of trees.
    A progress bar is displayed during the training process.
    """
    X = get_features(train_df)
    y = train_df['label'].astype(int)  # Convert labels to integers

    trees = []
    for i in tqdm(range(100), desc="Training trees"):
        # Set different random_state to ensure each tree selects different splitting features randomly
        clf = DecisionTreeClassifier(criterion='entropy', max_features=11, random_state=i)
        clf.fit(X, y)
        trees.append(clf)
    return trees

def predict_ensemble(trees, df):
    """
    Uses the ensemble model to predict labels for the input DataFrame.
    Each tree returns the predicted probabilities for each class, then the average probability is computed,
    and the class with the highest probability is chosen as the prediction.
    Returns an array representing the predicted label for each sample.
    """
    X = get_features(df)
    # Obtain the predicted probabilities from all trees; each element in the list has shape (n_samples, n_classes)
    prob_list = [clf.predict_proba(X) for clf in trees]
    
    # Compute the average of the predicted probabilities across all trees
    avg_prob = np.mean(prob_list, axis=0)
    
    # Retrieve the class labels (assuming all trees have the same classes_)
    classes = trees[0].classes_
    
    # For each sample, choose the class with the highest predicted probability
    predicted_labels = classes[np.argmax(avg_prob, axis=1)]
    return predicted_labels

def get_predicted_label_sequence(trees, df):
    """
    Generates a sequence of predicted labels by directly calling predict_ensemble,
    converts the result into a Pandas Series, and returns the predicted label sequence
    aligned with the input DataFrame's index.
    """
    pred_labels = predict_ensemble(trees, df)
    return pd.Series(pred_labels, index=df.index)

def evaluate_ensemble(trees, df):
    """
    Evaluates the classification accuracy of the ensemble model on the input DataFrame.
    """
    y_true = df['label'].astype(int)
    y_pred = predict_ensemble(trees, df)
    acc = accuracy_score(y_true, y_pred)
    return acc
