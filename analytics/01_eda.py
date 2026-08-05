import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Part - A: Profiling, Cleaning and the Data story
# --------------------------------------------------------------
# Task 1: Load, profile and save the titanic dataset
# --------------------------------------------------------------
print("=" * 50)
print("Load and Profile the 'titanic' dataset")
print("=" * 50)

try:
    # Load the dataset once and save it to .csv file
    df = sns.load_dataset("titanic")
    df.to_csv("titanic.csv", index = False)
    print("Titanic dataset from seaborn loaded and saved to titanic.csv (offline)\n")

    print("-----   Quick Summary   -----")
    print(df.info())

    print("\n-----   Statistical Summary   -----")
    print(df.describe(include= "all"))

    print("\n-----   Shape and Size   -----")
    print(f"No. of rows: {df.shape[0]}\nNo.of columns: {df.shape[1]}")

    print("\n-----   No. of NaN values in each column  -----")
    print(df.isnull().sum())
    print("\n-----   Percentage of missing values   -----")
    print((df.isnull().mean() * 100).round(3))

except Exception as e:
    print(f"Error: Task 1 (Load and profile dataset) failed {e}")

# --------------------------------------------------------------
# Task 2: Handling missing values
# --------------------------------------------------------------
print("=" * 50)
print("Handling missing values")
print("=" * 50)

try:
    # Make a copy of the original dataframe
    df_clean = df.copy()

    # Column "deck" has ~77% missing values. Imputing will result in fabricated information.
    # Rather than dropping, all the missing values can be considered as "Unknown" since it is a categorical type column
    # and "deck" is related to "pclass" & "survival". Otherwise drop the column
    df_clean["deck"] = df_clean["deck"].astype("object").fillna("Unknown")

    # Columns "embarked" and "embark_town" have <5% missing values.
    # Dropping those rows
    df_clean = df_clean.dropna(subset= ["embarked", "embark_town"])

    # Column "age" has ~20% missing values. Imputing the column with median since we have outliers (i.e., min age = 0.42)
    df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())

    print(f"\n-----   After missing value handling   -----")
    print(f"No.of rows: {df_clean.shape[0]}\nNo.of coulmns: {df_clean.shape[1]}")

    print("\n-----   Null values after missing value handling   -----")
    print(f"No.of NaN values in each column: \n{df_clean.isnull().sum()}")
    print(f"Percentage of missing values: \n{(df_clean.isnull().mean() * 100).round(3)}") 

except Exception as e:
    print(f"Error: Task 2 (Missing values handling) failed {e}")

# --------------------------------------------------------------
# Task 3: Univariate Analysis
# --------------------------------------------------------------
print("=" * 50)
print("Univariate Analysis")
print("=" * 50)

try:
    # Create a figure with four subplots
    fig, axes = plt.subplots(2, 2, figsize = (12, 9))

    # Plotting "age" histogram plot
    sns.histplot(df_clean["age"], kde= True, ax = axes[0,0])
    axes[0, 0].set_title("Age (Histogram plot)")

    # Plotting "age" boxplot
    sns.boxplot(x= df_clean["age"], ax= axes[0,1])
    axes[0,1].set_title("Age (Boxplot)")

    # Plotting "fare" histogram plot
    sns.histplot(df_clean["fare"], kde= True, ax= axes[1,0])
    axes[1,0].set_title("Fare (Histogram plot)")

    # Plotting "fare" boxplot
    sns.boxplot(x=df_clean["fare"], ax = axes[1,1])
    axes[1,1].set_title("Fare (Boxplot)")

    # Adjusting the plot spacing
    plt.tight_layout()

    # Save the figure
    plt.savefig("Univariate_analysis.png", dpi = 120)

    # Close the figure
    plt.close()
    print("Univariate_analysis.png saved successfully")



    # Define a function to detect no. of outliers
    def iqr_outlier_count(series):
        # Calculate q1 (25% quartile) 
        q1 = series.quantile(0.25)
        # Calculate q3 (75% quartile)
        q3 = series.quantile(0.75)

        # Calculate Inter-quartile range (IQR)
        iqr = q3 - q1

        # Calculate the lower limit
        lower = q1 - (1.5*iqr)

        # Calculate upper limit
        upper = q3 + (1.5*iqr)

        # Finding no. of outliers
        outliers = series[(series < lower) | (series > upper)]
        no_of_outliers = len(outliers)
        return no_of_outliers, lower, upper

    # Detect outliers in "age" column
    no_of_age_outliers, age_lower, age_upper = iqr_outlier_count(df_clean["age"])
    print(f"\nAge IQR Outliers: {no_of_age_outliers}")
    print(f"Limits: [{age_lower:.2f}, {age_upper:.2f}]")

    # Detect outliers in "fare" column
    no_of_fare_outliers, fare_lower, fare_upper = iqr_outlier_count(df_clean["fare"])
    print(f"\nFare IQR Outliers: {no_of_fare_outliers}")
    print(f"Limits: [{fare_lower:.2f}, {fare_upper:.2f}]")

    # Calculating fare (mean, median, mode)
    fare_mean = df_clean["fare"].mean()
    fare_median = df_clean["fare"].median()
    fare_mode = df_clean["fare"].mode()[0]

    print(f"\nFare Mean: {fare_mean:.2f}\nFare Median: {fare_median:.2f}\nFare Mode: {fare_mode:.2f}")

    # Determining the Skewness
    if fare_mean > fare_median > fare_mode:
        print(f"\nFare Distribution: Right-skewed (mean > meadian > mode).")
        print(f"Given mean ({fare_mean:.2f}) is well above median ({fare_median:.2f}), which is above ({fare_mode:.2f}).")
        print(f"A small number of very high fares are pulling the mean upward relative to the bulk of lower fare passengers.")
        print(f"Consistent with a right-skewed distribution.")

    elif fare_mean < fare_median < fare_mode:
        print(f"\nFare Distribution: Left-skewed (mean < median < mode).")

    else:
        print(f"\nFare distribution is approximately symmetric (no clear mean/median/mode order).")

except Exception as e:
    print(f"Error: Task 3 (Univariate Analysis - age, fare) failed {e}")

# --------------------------------------------------------------
# Task 4: Bivariate Analysis 
# --------------------------------------------------------------
print("\n" + "=" * 50)
print("Bivariate Analysis")
print("=" * 50)

try:
    # Survival rate analysis by sex
    print("\n-----   Survival rate by sex   -----")

    # Get all the unique genders
    unique_genders = df_clean["sex"].unique()

    for value in unique_genders:
        # Create a boolean filter mask for current sex
        mask = df_clean["sex"] == value
        # calculate the average survival rate
        survival_rate = df_clean.loc[mask, "survived"].mean()
        # Count no. of passengers that belong to that sex group
        total_count = mask.sum()

        print(f"Average Survival rate of {value} is {survival_rate:.3f}\nNo.of passengers survived in {value} are {total_count}")



    # Survival rate by pclass
    print("\n-----   Survival rate by pclass   -----")

    # Get all the unique passenger classes
    unique_pclass = df_clean["pclass"].unique()

    for pvalue in sorted(unique_pclass):
        # Create a boolean filter mask for current pclass
        pclass_mask = df_clean["pclass"] = pvalue
        # Calculate the average survival rate 
        pclass_survival_rate = df_clean.loc[pclass_mask, "survived"].mean()
        # No.of passengers that belong to that pclass
        pclass_total_count = pclass_mask.sum()

        print(f"Average Survival rate of {pvalue} is {pclass_survival_rate:.3f}\nNo.of passengers survived in {pvalue} are {pclass_total_count}")



    # Survival rate by sex and pclass together
    print("\n-----   Survival rate by sex and pclass   -----")

    for value in unique_genders:
        for pvalue in sorted(unique_pclass):
            # Create a boolean filter mask
            gender_pclass_mask = (df_clean["sex"] == value) & (df_clean["pclass"] == pvalue)
            # Calculate the average survival rate
            gender_pclass_survival_rate = df_clean.loc[gender_pclass_mask, "survived"].mean()
            # No. of passengers that belong
            gender_pclass_count = gender_pclass_mask.sum()

            print(f"Average Survival rate of {value} in {pvalue} is {gender_pclass_survival_rate:.3f}\nNo.of passengers survived are {gender_pclass_count} in {pvalue} who are of {value}")




    # Correlation Analysis
    # Correlation matrix restricted to below columns are (numeric columns)
    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    # Compute correlation matrix
    corr_matrix = df_clean[corr_cols].corr()

    # print correlation matrix
    print("\n-----   Correlation Matrix   -----")
    print(corr_matrix.round(3))

    # Create a Heatmap figure
    plt.figure(figsize= (7, 6))

    # plot heatmap
    sns.heatmap(corr_matrix, annot= True, fmt= ".2f", cmap= "coolwarm", center= 0)

    # Set the title name
    plt.title("Correlation Heatmap (survived, pclass, age, sibsp, parch, fare)")

    # Adjusting the plot spacing
    plt.tight_layout()

    # saving the figure
    plt.savefig("Correlation_Heatmap.png", dpi = 120)

    # close the figure
    plt.close()
    print("Correlation_Heatmap.png saved successfully")



    # Identify top 2 strongest off-diagonal correlations by absolute value
    # Create an empty list
    corr_pairs = []

    # Compare evrey pair
    for i in range(len(corr_cols)):
        for j in range(i+1, len(corr_cols)):
            # Store each pair
            corr_1 = corr_cols[i]
            corr_2 = corr_cols[j]

            corr_pairs.append((corr_1, corr_2, corr_matrix.loc[corr_1, corr_2]))

    # define a function that extract and convert correlational value to absolute value
    def get_value(item):
        # Get value at index 2 (correlation score)
        corr_value = item[2]
        # Return absolute value
        return abs(corr_value)

    # sort the list from largest to smallest value using "get_value" function
    corr_pairs_sorted = sorted(corr_pairs, key = get_value, reverse= True)

    # Top 2 off-diagonal correlations
    top_2 = corr_pairs_sorted[:2]

    print("\n-----   Top 2 strongest correlations (by Absolute value)   -----")
    for corr_1, corr_2, val in top_2:
        print(f"{corr_1} <--> {corr_2}: {val:.3f}")

    # Interpretation
    print(f"\nInterpretation: The strongest relationship between '{top_2[0][0]}' and '{top_2[0][1]}' with absolute value = {top_2[0][2]:.3f},"
          f"followed by '{top_2[1][0]}' and '{top_2[1][1]}' with absolute value = {top_2[1][2]:.3f}."
          f"These magnitudes indicate the degree to which each pair move together linearly across passengers."
          f"The sign indicates the direction of that relationship. +ve --> Both variables tend to increase or decrease together. -ve --> when one variable increases, the other variable tends to decrease.")

except Exception as e:
    print(f"Error: Task 4 (Bivariate Analysis) failed {e}")

# --------------------------------------------------------------
# Task 5: Multivariate "Data Story"
# --------------------------------------------------------------
print("=" * 50)
print("Multivariate Data story")
print("=" * 50)

try:
    # Chart 1: Survival rate by pclass and sex
    # Create a figure
    plt.figure(figsize= (7, 5))
    # create a bar chart
    sns.barplot(data= df_clean, x= "pclass", y= "survived", hue= "sex")
    # Add title
    plt.title("Chart 1: Survival rate by pclass and sex")
    # Add label
    plt.ylabel("Survival rate")
    # Adjusting the layout
    plt.tight_layout()
    # Save the figure
    plt.savefig("Survival_rate_by_pclass_and_sex.png", dpi = 120)
    # Close the figure
    plt.close()
    print("Survival_rate_by_pclass_and_sex.png saved successfully")
    # Interpretation
    print("\nChart 1 Interpretation: Across every pclass, women survived at far higher rates than men, and this gap holds regardless of class."
          "Survival for both genders declines moving from 1st to 3rd class, but the decline is more for women")



    # Chart 2: Age Distribution by survival
    # Create a figure
    plt.figure(figsize= (7, 5))
    # create a box plot
    sns.boxplot(data= df_clean, x= "survived", y= "age")
    # Add title
    plt.title("Chart 2: Age Distribution by Survival Outcome")
    # Add label
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    # Adjusting the layout
    plt.tight_layout()
    # Save the figure
    plt.savefig("Age_distribution_by_Survival_outcome.png", dpi = 120)
    # Close the figure
    plt.close()
    print("Age_distribution_by_Survival_outcome.png saved successfully")
    # Interpretation
    print("Chart 2 Interpretation: Survivors are slightly younger than non-survivors, with a lower median age. This is consistent with children being prioritized during evacuation," 
          "though the overlap between the two distributions shows the age alone is a weak predictor.")



    # Chart 3: fare vs age scatter plot 
    # Create a figure
    plt.figure(figsize= (7, 5))
    # create a scatter plot
    sns.scatterplot(data= df_clean, x= "age", y= "fare", hue= "survived", alpha= 0.6)
    # Add title
    plt.title("Chart 3: Fare vs Age scatter plot")
    # Adjusting the layout
    plt.tight_layout()
    # Save the figure
    plt.savefig("Fare_vs_Age_scatterplot.png", dpi = 120)
    # Close the figure
    plt.close()
    print("Fare_vs_Age_scatterplot.png saved successfully")
    # Interpretation
    print("Chart 3 Interpretation: Passengers paying higher fares survived more often, whereas non-survivors are more in lower fares band.")



    # Chart 4: Pair-plot across key numeric features
    # pair-plot columns
    pairplot_cols = ["survived", "pclass", "age", "fare"]
    # create a pair-plot
    pp = sns.pairplot(df_clean[pairplot_cols], hue= "survived", diag_kind= "hist")
    # Add title
    pp.fig.suptitle("Chart 4: Pairwise relationships (survived, pclass, age, fare)", y= 1.02)
    # Adjusting the layout
    plt.tight_layout()
    # Save the figure
    plt.savefig("pairplot.png", dpi = 120)
    # Close the figure
    plt.close()
    print("pairplot.png saved successfully")
    # Interpretation
    print("Chart 4 Interpretation: Survivors are more in lower class (ie., first class) and with higher fares."
          "The separation by age is much weaker, indicating that class and fare are stronger predictors of survival than age in this dataset.")

except Exception as e:
    print(f"Error: Task 5 (Multivariate data story) failed {e}")

# --------------------------------------------------------------
# Task 6: Exploratory Checks
# --------------------------------------------------------------
print("=" * 50)
print("Exploratory Checks")
print("=" * 50)

try:
    # Print the mean and standard deviation before standardization
    mean_std = df_clean[["age", "fare"]].agg(["mean", "std"])
    result = mean_std.round(4)
    print(f"\nBefore standardization:\n{result}")

    # create StandardScaler
    scaler = StandardScaler()
    # standardize the data
    standardize = scaler.fit_transform(df_clean[["age", "fare"]])
    # convert the standardized array into DataFrame
    standardize_df = pd.DataFrame(standardize, columns= ["age_z", "fare_z"])

    # Print the mean and standard deviation after standardization
    result_z = standardize_df.agg(["mean", "std"]).round(4)
    print(f"\nAfter standardization: \n{result_z}")

    # Create two plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # plot age before scaling
    sns.histplot(df_clean["age"], kde=True, ax=axes[0], color="steelblue")
    # plot age after scaling
    sns.histplot(standardize_df["age_z"], kde=True, ax=axes[0], color="orange")
    # Add title
    axes[0].set_title("Age: Before standardized (blue) vs After standardized (orange)")

    # Plot fare before scaling
    sns.histplot(df_clean["fare"], kde=True, ax=axes[1], color="steelblue")
    # Plot fare after scaling
    sns.histplot(standardize_df["fare_z"], kde=True, ax=axes[1], color="orange")
    # Add title
    axes[1].set_title("Fare: Before standardized (blue) vs After standardized (orange)")

    # Adjust the layout
    plt.tight_layout()
    # Save the figure
    plt.savefig("standardization_before_after.png", dpi=120)
    # Close the figure
    plt.close()
    print("Saved standardization_before_after.png")

    # --------------------------------------------------------------
    # Save the clean dataframe
    # --------------------------------------------------------------
    df_clean.to_csv("titanic_clean.csv", index= False)
    print("Saved successfully the clean Dataframe to 'titanic_clean.csv' file")

except Exception as e:
    print(f"Error: Task 6 (Exploratory checks (age, fare)) failed {e}")