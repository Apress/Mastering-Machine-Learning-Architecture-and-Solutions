import pandas as pd

# Sample data with different date formats
date_strings = ["2023-10-26", "10/26/2023", "Oct 26, 2023"]

# Convert to datetime objects with a specified format
dates = pd.to_datetime(date_strings, errors='coerce') #errors='coerce' will convert invalid parsing into NaT.

# Format the dates consistently (e.g., YYYY-MM-DD)
formatted_dates = dates.strftime('%Y-%m-%d')

print("Original Dates:", date_strings)
print("Formatted Dates:", formatted_dates)

