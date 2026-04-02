def download_dataset():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    file_name = "creditcard.csv"

    if not os.path.exists(file_name):
        print(f"Downloading dataset from {dataset_url}...")
        response = requests.get(dataset_url)
        if response.status_code == 200:
            with open(file_name, 'wb') as file:
                file.write(response.content)
            print(f"Dataset downloaded and saved as '{file_name}'.")
        else:
            raise Exception("Failed to download the dataset. Please check your internet connection.")
    else:
        print(f"Dataset '{file_name}' already exists. Skipping download.")


print("\nStep 2: Checking for dataset...")
download_dataset()
