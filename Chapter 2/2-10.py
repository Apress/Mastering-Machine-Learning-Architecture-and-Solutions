import pandas as pd
import matplotlib.pyplot as plt

# 1. Create sample data with datetime and sales
df = pd.DataFrame({
    'datetime': pd.date_range('2023-01-01', '2023-12-31', freq='D'),
    'sales': [100 + (i % 30) * 5 for i in range(365)]  # Synthetic seasonal pattern
})

# 2. Extract datetime features
df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday
df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour  # (Will be 0 since daily data)
df['week_of_year'] = df['datetime'].dt.isocalendar().week

# 3. Analyze trends
print("Average sales by day of week:")
print(df.groupby('day_of_week')['sales'].mean())

print("\nAverage sales by month:")
print(df.groupby('month')['sales'].mean())

# 4. Visualize monthly trend
df.groupby('month')['sales'].mean().plot(
    kind='bar',
    title='Seasonal Trend: Average Sales by Month',
    xlabel='Month',
    ylabel='Sales'
)
plt.show()
