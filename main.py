# This model predicts which geographical region a covid-19 mutation originates from using logistic regression
# Import required libraries
from Bio import SeqIO
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
from sklearn import model_selection, linear_model
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer  # NEW: Imputer to handle NaNs

file_path = "C:/Users/gunta/OneDrive - Queen's University/Second Year/COVID-19/SARS_CoV_2_sequences_global.fasta"

# Load sequences
sequences = [r for r in SeqIO.parse(file_path, 'fasta')]
reference = np.array(sequences[0])
mutations_per_seq = np.array([sum(np.array(s) != reference) for s in sequences])

# Plot mutation distribution
plt.hist(mutations_per_seq)
plt.xlabel('# mutations')
plt.ylabel('# sequences')
plt.show()

# Random sequence selection
min_number_of_mutations = 340
idx = np.random.choice(np.where(mutations_per_seq > min_number_of_mutations)[0])
print(f"Sequence {idx} has > {min_number_of_mutations} mutations!\n")
print(sequences[idx], '\n')
print("The sequence is composed of: ")
print(Counter(np.array(sequences[idx])))

# Count sequences with at least one 'N'
n_sequences_with_N = sum(['N' in s for s in sequences])
print(f"{n_sequences_with_N} sequences have at least 1 'N'!")

# Create a mutation feature matrix
mutation_df = pd.DataFrame()
n_bases_in_seq = len(sequences[0])

for location in tqdm.tqdm(range(n_bases_in_seq)):
    bases_at_location = np.array([s[location] for s in sequences])
    
    # Skip if no mutations
    if len(set(bases_at_location)) == 1:
        continue

    for base in ['A', 'T', 'G', 'C', '-']:
        feature_values = (bases_at_location == base).astype(float)  # Convert to float first
        feature_values[bases_at_location == 'N'] = np.nan  # Assign NaN        
        column_name = f"{location}_{base}"
        mutation_df[column_name] = feature_values

# Print feature matrix size
print(f"Size of matrix: {mutation_df.shape[0]} rows x {mutation_df.shape[1]} columns")

# Assign region labels based on sequence metadata
countries = [(s.description).split('|')[-1] for s in sequences]
countries_to_regions_dict = {
    'Australia': 'Oceania',
    'China': 'Asia',
    'Hong Kong': 'Asia',
    'India': 'Asia',
    'Nepal': 'Asia',
    'South Korea': 'Asia',
    'Sri Lanka': 'Asia',
    'Taiwan': 'Asia',
    'Thailand': 'Asia',
    'USA': 'North America',
    'Viet Nam': 'Asia'
}

# Assign regions, replacing missing values with 'NA'
regions = [countries_to_regions_dict.get(c, 'NA') for c in countries]

# Add labels **AFTER** filtering non-NA rows
mutation_df['label'] = regions

# Balance the dataset by sampling an equal number of sequences per region
balanced_df = mutation_df[mutation_df.label != 'NA'].drop_duplicates()
samples_north_america = balanced_df[balanced_df.label == 'North America']
samples_oceania = balanced_df[balanced_df.label == 'Oceania']
samples_asia = balanced_df[balanced_df.label == 'Asia']

# Balance classes
n = min(len(samples_north_america), len(samples_oceania), len(samples_asia))
balanced_df = pd.concat([samples_north_america.iloc[:n], 
                         samples_asia.iloc[:n], 
                         samples_oceania.iloc[:n]])

print("Number of samples in each region:", Counter(balanced_df['label']))

# Train a logistic regression model
lm = linear_model.LogisticRegression(
    multi_class="multinomial", max_iter=1000,
    fit_intercept=False, tol=0.001, solver='saga', random_state=42)

# Prepare training data
X = balanced_df.drop(columns=['label'])
Y = balanced_df['label']

# Handle missing values using an imputer
imputer = SimpleImputer(strategy="most_frequent")  # NEW: Replace NaNs with most frequent value
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)

# Train the model
lm.fit(X_train, Y_train)

# Make predictions
Y_pred = lm.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(Y_test, Y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Compute confusion matrix
confusion_mat = confusion_matrix(Y_test, Y_pred)

# Convert to DataFrame for better readability
confusion_df = pd.DataFrame(confusion_mat, 
                            index=[f"{c} (True)" for c in lm.classes_],
                            columns=[f"{c} (Predicted)" for c in lm.classes_])

# Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_df)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
