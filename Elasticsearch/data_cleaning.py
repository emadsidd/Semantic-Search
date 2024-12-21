import pandas as pd

# Load data
df = pd.read_csv('testing_sample_10000.csv', sep=',', encoding='utf-8-sig')
df.columns = df.columns.str.strip()
print("Columns:", df.columns)

with open('testing_sample_10000.csv', 'r', encoding='utf-8-sig') as f:
    first_line = f.readline()
    print("First line:", repr(first_line))

# Record the number of rows before dropping duplicates
initial_count = len(df)

# Drop duplicates based on parent_asin
df_unique = df.drop_duplicates(subset='parent_asin', keep='first')

# Record the number of rows after dropping duplicates
final_count = len(df_unique)

# Print out the counts to see if duplicates were removed
print(f"Initial number of rows: {initial_count}")
print(f"Number of rows after removing duplicates: {final_count}")

# Convert to a list of dicts
documents = df_unique.to_dict(orient='records')
# After dropping duplicates:
df_unique.to_csv('testing_sample_10000_unique.csv', index=False)

