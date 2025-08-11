import os
import requests
import gzip
import shutil

# --- Configuration ---
# The base URL for the IMDb datasets.
IMDB_DATASETS_URL = "https://datasets.imdbws.com/"

# A list of the specific dataset files we want to download for this project.
# title.basics: Contains movie titles, types, years, runtime, genres. CORE FILE.
# title.principals: Connects titles to people (actors, directors) via nconst. CORE FILE.
# name.basics: Contains person details (name, birth/death year) for each nconst. CORE FILE.
# title.crew: Contains director and writer nconsts for each title. Useful for director-focused analysis.
FILES_TO_DOWNLOAD = [
    "title.basics.tsv.gz",
    "title.principals.tsv.gz",
    "name.basics.tsv.gz",
    "title.crew.tsv.gz",
    "title.akas.tsv.gz",
    "title.ratings.tsv.gz", 
]

# The directory where we will store the raw, unzipped data.
# This directory is relative to the project's root folder.
RAW_DATA_DIR = "data/raw_imdb"


def download_and_unzip_imdb_data():
    """
    Downloads and unzips specified IMDb dataset files.
    
    This function performs the following steps:
    1. Ensures the target directory for raw data exists.
    2. Iterates through a predefined list of IMDb dataset files.
    3. For each file, it checks if the unzipped version already exists to prevent re-downloading.
    4. If not present, it downloads the gzipped file.
    5. It then unzips the file into the target directory.
    6. Finally, it removes the downloaded gzipped file to save space.
    """
    print("--- Starting IMDb Data Download and Unzip Process ---")

    # 1. Create the target directory if it doesn't exist.
    # The 'exist_ok=True' argument prevents an error if the directory is already there.
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    print(f"Ensured output directory exists: {RAW_DATA_DIR}")

    # 2. Loop through each file in our list.
    for filename_gz in FILES_TO_DOWNLOAD:
        # Construct the full URL for the file to download.
        url = f"{IMDB_DATASETS_URL}{filename_gz}"
        
        # Define the local file paths for both the gzipped and the final unzipped file.
        local_gz_path = os.path.join(RAW_DATA_DIR, filename_gz)
        # The final filename will be the same but without the '.gz' extension.
        final_tsv_path = local_gz_path[:-3]

        # 3. Check if the final unzipped file already exists.
        if os.path.exists(final_tsv_path):
            print(f"'{final_tsv_path}' already exists. Skipping download.")
            continue # Move to the next file in the loop.

        # 4. Download the file.
        print(f"Downloading '{filename_gz}'...")
        try:
            # We use stream=True to handle large files efficiently without loading them all into memory at once.
            with requests.get(url, stream=True) as r:
                r.raise_for_status() # This will raise an error if the download fails (e.g., 404 Not Found).
                with open(local_gz_path, 'wb') as f:
                    # Write the file to disk in chunks.
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Successfully downloaded to '{local_gz_path}'.")

            # 5. Unzip the file.
            print(f"Unzipping '{local_gz_path}'...")
            with gzip.open(local_gz_path, 'rb') as f_in:
                with open(final_tsv_path, 'wb') as f_out:
                    # shutil.copyfileobj is an efficient way to copy file contents.
                    shutil.copyfileobj(f_in, f_out)
            print(f"Successfully unzipped to '{final_tsv_path}'.")

        except requests.exceptions.RequestException as e:
            print(f"ERROR: Failed to download {url}. Reason: {e}")
            # If download fails, we skip to the next file.
            continue
        finally:
            # 6. Clean up: remove the gzipped file if it exists.
            if os.path.exists(local_gz_path):
                os.remove(local_gz_path)
                print(f"Cleaned up (removed) '{local_gz_path}'.")
        
        print("-" * 20) # Separator for clarity in the console output.

    print("--- Data Download and Unzip Process Finished ---")

# This is a standard Python convention. 
# The code inside this 'if' block will only run when you execute the script directly
# (e.g., 'python scripts/01_download_data.py'), not when it's imported by another script.
if __name__ == "__main__":
    download_and_unzip_imdb_data()