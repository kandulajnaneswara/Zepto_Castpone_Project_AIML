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
from imblearn.over_sampling import SMOTE



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
print()
print("=" * 50)
print("Task 7: Stratified train - test split")
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
print()
print("=" * 50)
print("Task 8: Preprocessing - fit on train data only")
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
    print("Preprocessing on training dataset is completed.")

except Exception as e:
    print(f"Error: Task 8 (Preprocessing - fit on Train data only) failed {e}")


# --------------------------------------------------------------
# Task 9: Train three classifiers on identical split
# -------------------------------------------------------------- 
print()
print("=" * 50)
print("Task 9: Train three classifiers on identical split")
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
print()
print("=" * 50)
print("Task 10: Evaluate all the three models")
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

# --------------------------------------------------------------
# Task 11: Imbalance handling comparision (baseline vs class_weight vs SMOTE)
# --------------------------------------------------------------
print()
print("=" * 50)
print("Task 11: Imbalance handling comparision (baseline vs class_weight vs SMOTE)")
print("=" * 50)

try:
    # Display the class distribution
    print("Class distribution for training set: ")
    print(y_train.value_counts())

    # Create a preprocessing pipeline
    imablance_preprocessor = ColumnTransformer(transformers= [("numeric", Pipeline(steps= [("imputer", SimpleImputer(strategy= "median")),
                                                                                           ("scaler", StandardScaler())]),
                                                                                    numerical_features),
                                                            ("category", Pipeline(steps= [("imputer", SimpleImputer(strategy= "most_frequent")),
                                                                                          ("onehot", OneHotEncoder(handle_unknown= "ignore"))]),
                                                                                    categorical_features)])
    # Fit_transform the preprocessing on training data
    X_train_proc = imablance_preprocessor.fit_transform(X_train)
    # Transform the test data
    X_test_proc = imablance_preprocessor.transform(X_test)

    # Create list to store results
    imbalance_results = []

    # Strategy A: Baseline (no imbalance handling)
    # Create a baseline model
    baseline = LogisticRegression(max_iter= 1000, random_state= 42)
    # Train the baseline model
    baseline.fit(X_train_proc, y_train)
    # Predict test data
    pred_baseline = baseline.predict(X_test_proc)
    # Save evaluation metrics
    imbalance_results.append({"Strategy": "Baseline (no imbalance handling)",
                              "Precision score": precision_score(y_test, pred_baseline),
                              "Recall score": recall_score(y_test, pred_baseline),
                              "F1 score": f1_score(y_test, pred_baseline)})


    # Strategy B: class_weight = "balanced"
    # Create the model
    balanced = LogisticRegression(max_iter= 1000, class_weight= "balanced", random_state= 42)

    # Train the model
    balanced.fit(X_train_proc, y_train)
    # Predict the test data
    pre_balanced = balanced.predict(X_test_proc)
    # Save evaluation metrics
    imbalance_results.append({"Strategy": "class_weight = 'balanced'",
                              "Precision score": precision_score(y_test, pre_balanced),
                              "Recall score": recall_score(y_test, pre_balanced),
                              "F1 score": f1_score(y_test, pre_balanced)})


    # Strategy C: SMOTE - applied only to the training data to avoid leakages
    # Create a SMOTE object
    smote = SMOTE(random_state= 42)
    # Apply SMOTE to the training data
    X_train_smote, y_train_smote = smote.fit_resample(X_train_proc, y_train)
    # Create SMOTE model
    smote_model = LogisticRegression(max_iter= 1000, random_state= 42)
    # Train on the resampled data
    smote_model.fit(X_train_smote, y_train_smote)
    # Predict the original test dataset
    pred_smote = smote_model.predict(X_test_proc)
    # Save evaluation metrics
    imbalance_results.append({"Strategy": "SMOTE (Train dataset only)",
                              "Precision score": precision_score(y_test, pred_smote),
                              "Recall score": recall_score(y_test, pred_smote),
                              "F1 score": f1_score(y_test, pred_smote)})


    # Create a comparision table for the three strategies
    imbalance_df = pd.DataFrame(imbalance_results).set_index("Strategy")

    # Print the imbalance dataframe
    print(f"\n-----   Comparision of Imbalanced Strategies   -----")
    print(f"{imbalance_df.round(3)}")

    # Finding Best strategy
    best_strategy = imbalance_df["F1"].idxmax()
    print(f"Conclusion: '{best_strategy}' produced the highest F1 score ({imbalance_df.loc[best_strategy, 'F1']:.3f}) among the three strategies.")

except Exception as e:
    print(f"Error: Task 11 (Imbalance handling comparision) failed {e}")


# --------------------------------------------------------------
# Task 12: Hyperparameter Tuning (GridSearchCV) 
# --------------------------------------------------------------
print()
print("=" * 50)
print("Task 12: Hyperparameter Tuning (GridSearchCV)")
print("=" * 50)

try:
    # Create a Random Forest pipeline
    rf_pipeline = Pipeline(steps= [("prepocessor", preprocessor),
                                   ("classifier", RandomForestClassifier(oob_score= True, random_state= 42, bootstrap= True))])

    # Define Hyperparameter grid
    param_grid = {"classifier__n_estimators": [100, 200, 300],
                  "classifier__max_depth": [None, 5, 10],
                  "classifier__max_features": ["sqrt", "log2"]}

    # Create GridSearchCV
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv= 5, scoring= "f1", n_jobs= -1)
    # Train the GridSearchCV
    grid_search.fit(X_train, y_train)
    # Print the best parameters and cross-validation score
    best_params = grid_search.best_params_
    grid_search_f1 = grid_search.best_score_

    print(f"Best Parameters: {best_params}")
    print(f"Best GridSearchCV F1 score: {grid_search_f1:.3f}")

    # Retrieve the best model
    best_rf_pipeline = grid_search.best_estimator_
    # Extract OOB score
    oob_score = best_rf_pipeline.named_steps["classifier"].oob_score_
    print(f"OOB Score (best estimator, oob_score = True): {oob_score:.3f}")

except Exception as e:
    print(f"Error: Task 12 Hyperparameter Tuning (GridSearchCV) failed {e}")

# --------------------------------------------------------------
# Task 13: Regression side-task
# --------------------------------------------------------------
print()
print("=" * 50)
print("Task 13: Regression task - predict fare")
print("=" * 50)

try:
    # Define regression features
    Reg_Features = ["pclass", "sex", "age", "sibsp", "parch", "embarked"]
    # Separate numerical and categorical columns
    Reg_Numeric = ["pclass", "age", "sibsp", "parch"]
    Reg_Categorical = ["sex", "embarked"]

    # Create input data (features matrix)
    X_reg  =df[Reg_Features].copy()
    # Create target variable
    y_reg = df["fare"].copy()
    # Splitting into Train and Test data
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size= 0.2, random_state= 42)
    # Create preprocessing pipeline
    reg_preprocessor = ColumnTransformer(transformers= [("numeric", Pipeline(steps= [("imputer", SimpleImputer(strategy= "median")),
                                                                                     ("scaler", StandardScaler())]),
                                                                            Reg_Numeric),
                                                        ("category", Pipeline(steps= [("imputer", SimpleImputer(strategy= "most_frequent")),
                                                                                      ("onehot", OneHotEncoder(handle_unknown= "ignore")),]),
                                                                            Reg_Categorical)])
    # Create Regression pipeline
    reg_pipeline = Pipeline(steps= [("preprocessor", reg_preprocessor),
                                    ("regressor", LinearRegression())])

    # Train the Regression model
    reg_pipeline.fit(X_reg_train, y_reg_train)
    # Predict fare
    y_reg_pred = reg_pipeline.predict(X_reg_test)

    # Calculating Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    # Calculating Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    # Calculating Coefficient of Determination (R2)
    r2 = r2_score(y_reg_test, y_reg_pred)

    # No. of test observations
    n = len(y_reg_test)
    # Counting the transformed features
    p = reg_pipeline.named_steps["preprocessor"].transform(X_reg_test).shape[1]
    # Calculating Adjusted R2
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # Print evaluation metrics
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")
    print(f"Adjusted R2: {adjusted_r2:.3f}")

    # Compute residuals
    residuals = y_reg_test - y_reg_pred

    # Creating Residual Plot
    # Create the figure
    plt.figure(figsize= (7, 5))
    # Create scatter plot
    plt.scatter(y_reg_pred, residuals, alpha= 0.6)
    # Draw zero error line
    plt.axhline(0, color = 'red', linestyle= "--")
    # Label X axis
    plt.xlabel("Predicted Fare")
    # Label y axis
    plt.ylabel("Residual (Actual - Predicted)")
    # Add title
    plt.title("Residual Plot - Fare (Linear Regression)")
    # Adjust the layout
    plt.tight_layout()
    # Save the figure
    plt.savefig("residual_plot.png", dpi= 120)
    # Close the figure
    plt.close()
    print("'residual_plot.png' saved successfully")

    # Finding the median prediction
    pred_median = np.median(y_reg_pred)
    # Computing residual variability range
    residual_low = residuals[y_reg_pred < pred_median]
    residual_high = residuals[y_reg_pred >= pred_median]
    # Standard deviation of the residual_low
    residual_std_low = residual_low.std()
    # Standard deviation of the residual_high
    residual_std_high = residual_high.std()
    # Compare both the residual values
    print(f"\nResidual standard deviation (low predicted fare): {residual_std_low:.2f}")
    print(f"Residual standard deviation (high predicted fare): {residual_std_high:.2f}")

except Exception as e:
    print(f"Error: Task 13 (Regression task - predict fare) failed {e}")

# --------------------------------------------------------------
# Task 14: Model Comparision table
# --------------------------------------------------------------
print()
print("=" * 50)
print("Task 14: Model Comparision Table")
print("=" * 50)

try:
    print("\n-----   Classification models   -----")
    # Print comparision table
    print(evaluation_df.round(3))
    # Create a regression summary table
    regression_summary = pd.DataFrame([{"MAE": mae,
                                        "RMSE": rmse,
                                        "R2": r2,
                                        "Adjusted R2": adjusted_r2}], index= ["Linear Regression (fare)"])

    print("\n-----   Regression Model Summary   -----")
    # Print the regression table
    print(regression_summary.round(3))
    # Best classifier
    best_classifier = evaluation_df["F1 score"].idxmax()
    # Retrieve models metrics
    best_row = evaluation_df.loc[best_classifier]
    print(f"\nRecommendation: of the three classifiers, '{best_classifier}' is the strongest classifier to deploy"
          f"with F1 = {best_row['F1 score']:.3f}, accuracy = {best_row['Accuracy score']:.3f} and AUC = {best_row['AUC score']:.3f} on the test set."
          f"It balances the precision {best_row['Precision score']:.3f} and recall {best_row["Recall score"]:.3f} better than the alternatives.")

except Exception as e:
    print(f"Error: Task 14 (Model comparision table) failed {e}")

# --------------------------------------------------------------
# Task 15: Save and Reload full pipeline (preprocessing + estimator)
# --------------------------------------------------------------
print()
print("=" * 50)
print("Task 15: Save and Reload full pipeline")
print("=" * 50)

try:
    # Selecting best classifier by F1 score
    full_pipeline = training_models[best_classifier]
    # Save the pipeline
    joblib.dump(full_pipeline, "titanic_best_pipeline.joblib")
    print(f"Saved Full pipeline ('{best_classifier}') to 'titanic_best_pipeline.joblib' successfully")
    # Reload the saved pipeline
    reloaded_pipeline = joblib.load("titanic_best_pipeline.joblib")
    # Select the raw sample data
    sample_data = X_test.iloc[:5]
    # Predict using Original full pipeline
    original_pred = full_pipeline.predict(sample_data)
    # Predict using reloaded pipeline
    reloaded_pred = reloaded_pipeline.predict(sample_data)
    # Compare both predictions
    print(f"Prediction match: {np.array_equal(original_pred, reloaded_pred)}")

except Exception as e:
    print(f"Error: Task 15 (Save and reload full pipeline) failed {e}")