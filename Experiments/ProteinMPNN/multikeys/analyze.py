import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Read and concatenate all the CSV files
csv_files = glob.glob('multikeys_results_*.csv')
df_list = [pd.read_csv(file) for file in csv_files]

# Combine all dataframes into one
data = pd.concat(df_list, ignore_index=True)

# Step 2: Annotate the DataFrame with an additional column
# Add a column 'is_true_key' which is True if true_key == detect_key, else False
data['is_true_key'] = data['true_key'] == data['detect_key']

# Step 3: Calculate the final_score and p-value for each sequence
# Multiply the watermark_score per token by the total number of tokens (300)
data['final_score'] = data['watermark_score'] * 300

# Calculate p-value as np.exp(-final_score)
data['p_value'] = np.exp(-data['final_score'])

# Step 4: Evaluate the false positive ratio
# Define the threshold for p-value
threshold = 0.001

# Total number of negative cases (where true_key != detect_key)
total_negative_cases = len(data[data['is_true_key'] == False])

# Number of false positives (negative cases where p_value < threshold)
false_positives = data[
    (data['is_true_key'] == False) & 
    (data['p_value'] < threshold)
]
num_false_positives = len(false_positives)

# Calculate the false positive ratio
if total_negative_cases > 0:
    false_positive_ratio = num_false_positives / total_negative_cases
else:
    false_positive_ratio = 0

# Step 5: Evaluate the false negative rate
# Total number of positive cases (where true_key == detect_key)
total_positive_cases = len(data[data['is_true_key'] == True])

# Number of false negatives (positive cases where p_value >= threshold)
false_negatives = data[
    (data['is_true_key'] == True) & 
    (data['p_value'] >= threshold)
]
num_false_negatives = len(false_negatives)

# Calculate the false negative rate
if total_positive_cases > 0:
    false_negative_rate = num_false_negatives / total_positive_cases
else:
    false_negative_rate = 0

# Plot watermark_score distribution using violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(x='is_true_key', y='final_score', data=data)
plt.xlabel('True Key Matches Detection Key')
plt.ylabel('Watermark Score per Seq')
plt.title('Distribution of Watermark Scores')
plt.savefig('watermark_score_violin_plot.png')
plt.close()

# Print the results
print(f"Total negative cases (true_key != detect_key): {total_negative_cases}")
print(f"Number of false positives (p_value < {threshold}): {num_false_positives}")
print(f"False positive ratio: {false_positive_ratio:.6f}")

print(f"Total positive cases (true_key == detect_key): {total_positive_cases}")
print(f"Number of false negatives (p_value >= {threshold}): {num_false_negatives}")
print(f"False negative rate: {false_negative_rate:.6f}")

# Report watermark scores of false negative samples
print("Watermark Scores of False Negative Samples:")
print(false_negatives['final_score'])
