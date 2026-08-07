import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



# Part - B: Predictive modeling, continuing from the cleaned data
# Load the clean Dataframe "titanic_clean.csv" obtained in 01_eda.py
df = pd.read_csv("titanic_clean.csv")
print(f"Successfully loaded 'titanic_clean.csv':\nTotal rows: {df.shape[0]}\nTotal columns: {df.shape[1]}")

# Selecting the features and Target for the model
target = "survived"
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]

# Defining X and y with a Dataframe copy
X = df[features].copy()
y = df[target].copy()

# Defining categorical columns and numerical columns
categorical_features = ["sex", "embarked"]
numerical_features = ["pclass", "age", "sibsp", "parch", "fare"]

# --------------------------------------------------------------
# Task 7: Startified train - test split
# --------------------------------------------------------------
print("=" * 50)
print("Stratified train - test split")
print("=" * 50)

try:
    # Display the target class distribution for passengers who survived and not-survived (in percentage)
    print("Target class balance: ")
    target_class_balance = (y.value_counts(normalize=True) * 100).round(4)
    print(f"Percentage of Passengers not survive: {target_class_balance[0]}")
    print(f"Percentage of Passengers survived: {target_class_balance[1]}")

    # Stratified train and test spliting
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify= y, test_size= 0.2, random_state= 42)

    print(f"\nTrain shape: {X_train.shape}\nTest shape: {X_test.shape}")
    print("Train class balance: ")
    print((y_train.value_counts(normalize= True) * 100).round(4))
    print("Test class balance: ")
    print((y_test.value_counts(normalize= True) * 100).round(4))

    # Justification
    print("\nJustification: \nThe target class 'survived' is imbalanced dataset with ~39% percentage of survived passengers and ~61% percentage of not survived passengers.")
    print("Imbalanced dataset will lead to inconsistent or biased model evaluation.")
    print("The Stratify Train - Test split guarantees both train dataset & test dataset splits with original data ratio i.e., (~61/39 ratio)")
    
except Exception as e:
    print(f"Error: Task 7 (Stratified train - test split) failed {e}")


# --------------------------------------------------------------
# Task 8: Preprocessing - fit on training data only (ColumnTransformer)
# --------------------------------------------------------------
print("=" * 50)
print("Preprocessing - fit on train data only")
print("=" * 50)

try:
    # Preprocessing pipeline for numerical features
    # If Numeric column contains any NaN values, they should be replaced with median of that column 
    # and then standardize respective column
    numeric_pipeline = Pipeline(steps= [("imputer", SimpleImputer(strategy= "median")),
                                        ("scaler", StandardScaler())])

    # Preprocessing pipeline for categorical features
    # If Category column contains any missing values, they should be replaced with most common value
    # and perform one-hot encoding
    category_pipeline = Pipeline(steps= [("imputer", SimpleImputer(strategy= "most_frequent")),
                                         ("onehot", OneHotEncoder(handle_unknown= "ignore"))])

    # Combine both numeric and category pipelines with ColumnTransformer
    preprocessor = ColumnTransformer(transformers= [("numeric", numeric_pipeline, numerical_features),
                                                    ("category", category_pipeline, categorical_features)])

except Exception as e:
    print(f"Error: Task 8 (Preprocessing - fit on Train data only) failed {e}")


# --------------------------------------------------------------
# Task 9: Train three classifiers on identical split
# -------------------------------------------------------------- 
print("=" * 50)
print("Train three classifiers on identical split")
print("=" * 50)

try:
    # Create a dictionary with three models
    models = {"Logistic Regression": LogisticRegression(max_iter= 1000, random_state= 42),
              "Decision Tree": DecisionTreeClassifier(random_state= 42),
              "Random Forest": RandomForestClassifier(random_state= 42)}

    # Create dictionary for training models
    training_models = {}
    # Loop through every model and train the models
    for name, model in models.items():
        # Create a pipeline using previous ColumnTransformer preprocessor
        pipe = Pipeline(steps= [("preprocessor", preprocessor),
                                ("classifier model", model)])
        # Train pipeline
        # fit on training data
        pipe.fit(X_train, y_train)
        # Save the trained pipeline
        training_models[name] = pipe
        print(f"Trained model: {name}")

    # Visualize the Decision Tree
    # retrieve the trained decision tree
    dt_pipeline = training_models["Decision Tree"]
    # Extract the decision tree classifier model only
    dt_model = dt_pipeline.named_steps["classifier model"]
    # Get feature names after preprocessing
    dt_features = dt_pipeline.named_steps["preprocessor"].get_feature_names_out()

    # Plot the decision tree
    # Create a figure
    plt.figure(figsize= (20, 10))
    # Visualize the trained decision tree
    plot_tree(dt_model, feature_names= dt_features, class_names= ["not survived", "survived"], filled= True, max_depth= 3, fontsize= 8)
    # Add the title
    plt.title("Decision Tree Plot")
    # Adjust layout
    plt.tight_layout()
    # Save the figure
    plt.savefig("decision_tree_plot.png", dpi= 120)
    # Close the figure
    plt.close()
    print("'decision_tree_plot.png' saved successfully")

except Exception as e:
    print(f"Error: Task 9 (Train three classifiers on the identical split) failed {e}")



# --------------------------------------------------------------
# Task 10: Evaluate all three models 
# --------------------------------------------------------------
print("=" * 50)
print("Evaluate all the three models")
print("=" * 50)

try:
    # Create a list to store the evaluation results
    evaluation_results = []
    # Create a figure for the ROC curves
    plt.figure(figsize= (7, 6))
    # Loop through every trained model
    for name, pipe in training_models.items():
        # Predict the class labels
        y_pred = pipe.predict(X_test)
        # Predict class probabilities (survived class)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        # Compute the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # Compute accuracy
        acc = accuracy_score(y_test, y_pred)
        # Compute precision
        prec = precision_score(y_test, y_pred)
        # Compute Recall
        rec = recall_score(y_test, y_pred)
        # Compute F1 Score
        f1 = f1_score(y_test, y_pred)
        # Compute AUC
        auc = roc_auc_score(y_test, y_pred)

        print(f"\n-----   {name}   -----")
        print(f"Confusion matrix: \n{cm}")
        print(f"Accuracy score: {acc:.3f}")
        print(f"Precision score: {prec:.3f}")
        print(f"Recall score: {rec:.3f}")
        print(f"F1 score: {f1:.3f}")
        print(f"AUC score: {auc:.3f}")

        # Compute ROC curve cordinates
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        # Plot the roc curve
        plt.plot(fpr, tpr, label= f"{name}      (AUC = {auc:.3f}")
        # Saving evaluation results 
        evaluation_results.append({"Model": name,
                                   "Accuracy score": acc,
                                   "Precision score": prec,
                                   "Recall score": rec,
                                   "F1 score": f1,
                                   "AUC score": auc})

    # Draw the diagonal line from (0,0) to (1,1) in ROC curve
    plt.plot([0, 1], [0, 1], linestyle= "--", color= "gray", label= "Chance")
    # Label X and Y axes
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    # Add a title
    plt.title("ROC Curves -- All three Classifiers")
    # Display legend
    plt.legend()
    # Adjust layout space
    plt.tight_layout()
    # Save the figure
    plt.savefig("roc_curves.png", dpi= 120)
    # Close the figure
    plt.close()
    print("'roc_curves.png' saved successfully")

    # Create a comparison table for all the three classifiers
    evaluation_df = pd.DataFrame(evaluation_results).set_index("Model")
    print(f"\n-----   Comparision of all three classifier models   -----")
    print(f"{evaluation_df.round(3)}")   

except Exception as e:
    print(f"Error: Task 10 (Evaluate all the three models) failed {e}")
