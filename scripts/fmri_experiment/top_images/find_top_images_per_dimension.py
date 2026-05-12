import numpy as np
import pandas as pd
import os
from os.path import join as pjoin, dirname
import sys

# Add the project root to the Python path (script lives in scripts/fmri_experiment/top_images/)
project_root = dirname(dirname(dirname(dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

try:
    from closedloop.data import get_thingsimages_fnames
except ImportError:
    print("Error: Could not import 'get_thingsimages_fnames' from 'src.ffadims.data'.")
    print("Make sure the script is run from the project root or the path is correctly set.")
    sys.exit(1)


def find_top_images(embedding_path, labels_path, n_top=6):
    """
    Finds the top N images for each dimension in an embedding and returns a table.

    Args:
        embedding_path (str): Path to the embedding file (.npy). Expected shape (n_dimensions, n_images).
        labels_path (str): Path to the text file containing dimension labels (one per line).
        n_top (int): Number of top images to find for each dimension.

    Returns:
        pd.DataFrame: DataFrame with dimensions as columns and top image filenames as rows.
                      Returns None if files cannot be loaded or dimensions mismatch.
    """
    # --- Load Data ---
    try:
        embedding = np.load(embedding_path)
        print(f"Loaded embedding from {embedding_path} with shape: {embedding.shape}")
    except FileNotFoundError:
        print(f"Error: Embedding file not found at {embedding_path}")
        # Try the potentially misspelled path
        embedding_path_alt = embedding_path.replace("group_avg.npy", "gropu_avg.npy")
        try:
            embedding = np.load(embedding_path_alt)
            print(f"Loaded embedding from {embedding_path_alt} with shape: {embedding.shape}")
            embedding_path = embedding_path_alt # Update path if alternate works
        except FileNotFoundError:
            print(f"Error: Also could not find embedding file at {embedding_path_alt}")
            return None
    except Exception as e:
        print(f"Error loading embedding file {embedding_path}: {e}")
        return None

    try:
        image_fnames = get_thingsimages_fnames()
        print(f"Loaded {len(image_fnames)} image filenames.")
    except Exception as e:
        print(f"Error loading image filenames: {e}")
        return None

    try:
        with open(labels_path, 'r') as f:
            dim_labels = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(dim_labels)} dimension labels from {labels_path}.")
    except FileNotFoundError:
        print(f"Warning: Dimension labels file not found at {labels_path}. Using default dimension indices.")
        dim_labels = [f"Dim_{i+1}" for i in range(embedding.shape[0])]
    except Exception as e:
        print(f"Error loading dimension labels file {labels_path}: {e}")
        print("Using default dimension indices.")
        dim_labels = [f"Dim_{i+1}" for i in range(embedding.shape[0])]

    # --- Validate Shapes ---
    n_dims, n_images = embedding.shape

    if n_images != len(image_fnames):
        print(f"Error: Number of images in embedding ({n_images}) does not match number of filenames ({len(image_fnames)}).")
        # If the embedding was (images, dims), transpose it
        if embedding.shape[0] == len(image_fnames):
             print("Transposing embedding to match (dims, images) format.")
             embedding = embedding.T
             n_dims, n_images = embedding.shape
        else:
             print("Cannot resolve dimension mismatch.")
             return None


    if n_dims != len(dim_labels):
        print(f"Warning: Number of dimensions in embedding ({n_dims}) does not match number of labels ({len(dim_labels)}). Using default labels for columns.")
        dim_labels = [f"Dim_{i+1}" for i in range(n_dims)]


    # --- Find Top Images ---
    top_images_dict = {}
    for i in range(n_dims):
        dim_values = embedding[i, :]
        # Get indices of the top N values (indices of largest values)
        top_indices = np.argsort(dim_values)[-n_top:][::-1] # Sort ascending, take last N, reverse
        # Get corresponding image filenames
        top_fnames = image_fnames[top_indices]
        # Extract only the filename part
        top_fnames_base = [os.path.basename(f) for f in top_fnames]
        top_images_dict[dim_labels[i]] = top_fnames_base

    # --- Create DataFrame ---
    # Create index for ranks 1 to n_top
    rank_index = [f"Top {j+1}" for j in range(n_top)]
    results_df = pd.DataFrame(top_images_dict, index=rank_index)

    return results_df

if __name__ == "__main__":
    # Default paths to the deposited embedding and dimension labels in this repo.
    EMBEDDING_FILE = "data/embedding/emb_filtered.npy"
    LABELS_FILE = "data/embedding/dimension_labels.txt"
    N_TOP_IMAGES = 6

    # Construct absolute paths from project root
    abs_embedding_path = os.path.join(project_root, EMBEDDING_FILE)
    abs_labels_path = os.path.join(project_root, LABELS_FILE)


    results_table = find_top_images(abs_embedding_path, abs_labels_path, N_TOP_IMAGES)

    if results_table is not None:
        print("\n--- Top Images Per Dimension ---")
        # Configure pandas display options for better readability
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200) # Adjust width as needed
        print(results_table)

        # Optionally save to CSV (next to this script)
        output_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "top_images_per_dimension.csv")
        results_table.to_csv(output_csv_path)
        print(f"\nTable saved to {output_csv_path}")
    else:
        print("\nScript finished with errors.") 