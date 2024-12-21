import pandas as pd
import json

# Step 1: Load your dataset
# Adjust the file name, encoding, and other parameters as needed.
df = pd.read_csv('testing_sample_10000.csv', encoding='utf-8-sig')

# Step 2: Group by 'query' and aggregate 'parent_asin' into a list
query_to_asins = df.groupby('query')['parent_asin'].apply(list).to_dict()

# Step 4 & 5:  Save ground truth as JSON
with open('ground_truth.json', 'w', encoding='utf-8') as f:
    json.dump(query_to_asins, f, ensure_ascii=False, indent=2)

print("Ground truth mapping saved to ground_truth.json")
