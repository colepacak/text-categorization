import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data.
df = pd.read_csv('data/bbc-text.csv')
# Get the category counts to see how evenly represented each category is in the data.
fig = plt.figure(figsize=(8,4))
sns.barplot(x = df['category'].unique(), y = df['category'].value_counts())
plt.title('Category Counts for BBC Articles')
plt.savefig('figures/category-counts')
