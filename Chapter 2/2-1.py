import requests
import pandas as pd

# Define the API endpoint and parameters
api_url = "https://api.openweathermap.org/data/2.5/weather"
params = {
    'q': 'London',
    'appid': 'YOUR_API_KEY'
}

# Make the API request
response = requests.get(api_url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    # Convert the relevant data to a pandas DataFrame
    weather_data = pd.DataFrame([data])
    print(weather_data)
else:
    print(f"Failed to retrieve data: {response.status_code}")
