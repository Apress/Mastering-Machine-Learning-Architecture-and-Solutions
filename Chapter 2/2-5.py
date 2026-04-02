import random

def check_missing_values(data):
    missing_count = sum(1 for x in data if x is None)
    return missing_count

def check_data_range(data, min_val, max_val):
    out_of_range_count = 0
    for x in data:
        if x is not None and isinstance(x, (int, float)) and not (min_val <= x <= max_val):  # Check type *before* comparing
            out_of_range_count += 1
    return out_of_range_count

def check_data_types(data, expected_type):
    incorrect_type_count = 0
    for x in data:
        if x is not None and not isinstance(x, expected_type):
            incorrect_type_count += 1
    return incorrect_type_count

# Sample data
temperature_data = [25, 22, None, 28, 31, 150, 27, None, 29, "abc", 30.5]  # Includes missing, out-of-range, incorrect type, and float

# Performing data quality checks
missing_values = check_missing_values(temperature_data)
out_of_range_values = check_data_range(temperature_data, 0, 100)
incorrect_types = check_data_types(temperature_data, int)

print(f"Missing Values: {missing_values}")
print(f"Out of Range Values: {out_of_range_values}")
print(f"Incorrect Type Values: {incorrect_types}")
