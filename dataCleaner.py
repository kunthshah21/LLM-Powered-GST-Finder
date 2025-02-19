import pandas as pd
df = pd.read_csv("data.csv")

# Trim extra spaces at the beginning and end of text in all columns
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Remove rows where "Description of Goods" is exactly "[Omitted]" or "Omitted"
df = df[~df["Description of Goods"].isin(["[Omitted]", "Omitted"])]


df.to_csv("cleaned_data.csv", index=False)
print("Data cleaning complete. Saved as 'cleaned_data.csv'.")
