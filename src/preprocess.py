import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
from collections import Counter
import os

def preprocess_data(data_file, output_dir):
    """
    Preprocesses raw data for training machine learning models.

    Steps:
    1. Load the data from a CSV file.
    2. Handle missing values by dropping rows with NaN values.
    3. Encode categorical labels in the 'family_accession' column.
    4. Split the data into train, dev, and test sets based on class distribution.
    5. Save the processed datasets and metadata (e.g., label encoder, class weights).
    """

    # Step 1: Load the data
    print('Loading data...')
    data = pd.read_csv(data_file)

    # Step 2: Handle missing values
    print('Handling missing values...')
    data = data.dropna()

    # Step 3: Encode the 'family_accession' column to numeric labels
    print('Encoding labels...')
    label_encoder = LabelEncoder()
    data['class_encoded'] = label_encoder.fit_transform(data['family_accession'])

    # Save the label encoder for future use
    os.makedirs(output_dir, exist_ok=True)
    label_encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Label encoder saved to {label_encoder_path}")

    # Save the label mapping to a text file
    label_mapping_path = os.path.join(output_dir, 'label_mapping.txt')
    with open(label_mapping_path, 'w') as f:
        for class_label, class_name in enumerate(label_encoder.classes_):
            f.write(f"{class_label}: {class_name}\n")
    print(f"Label mapping saved to {label_mapping_path}")

    # Step 4: Split the data into train, dev, and test sets
    print('Splitting data into train, dev, and test sets...')
    train_indices = []
    dev_indices = []
    test_indices = []

    for cls, group in data.groupby('class_encoded'):
        indices = group.index.tolist()
        if len(indices) == 1:
            test_indices.extend(indices)
        elif len(indices) == 2:
            dev_indices.append(indices[0])
            test_indices.append(indices[1])
        elif len(indices) == 3:
            train_indices.append(indices[0])
            dev_indices.append(indices[1])
            test_indices.append(indices[2])
        else:
            train, temp = train_test_split(indices, test_size=0.4, random_state=42, stratify=group['class_encoded'])
            dev, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=data.loc[temp, 'class_encoded'])
            train_indices.extend(train)
            dev_indices.extend(dev)
            test_indices.extend(test)

    # Step 5: Create DataFrames for train, dev, and test sets
    train_df = data.loc[train_indices].reset_index(drop=True)
    dev_df = data.loc[dev_indices].reset_index(drop=True)
    test_df = data.loc[test_indices].reset_index(drop=True)

    # Step 6: Drop unused columns
    print('Dropping unused columns...')
    columns_to_drop = ['family_id', 'sequence_name']  # Adjust based on your dataset
    train_df = train_df.drop(columns=columns_to_drop, errors='ignore')
    dev_df = dev_df.drop(columns=columns_to_drop, errors='ignore')
    test_df = test_df.drop(columns=columns_to_drop, errors='ignore')

    # Step 7: Save the train, dev, and test datasets
    print('Saving datasets...')
    train_path = os.path.join(output_dir, 'train.csv')
    dev_path = os.path.join(output_dir, 'dev.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train dataset saved to {train_path}")
    print(f"Dev dataset saved to {dev_path}")
    print(f"Test dataset saved to {test_path}")

    # Step 8: Calculate class weights from the training set
    print('Calculating class weights...')
    class_counts = Counter(train_df['class_encoded'])
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # Normalize weights
    max_weight = max(class_weights.values())
    class_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}

    # Save class weights
    class_weights_path = os.path.join(output_dir, 'class_weights.json')
    with open(class_weights_path, 'w') as f:
        f.write(str(class_weights))
    print(f"Class weights saved to {class_weights_path}")

    print('Preprocessing complete!')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess protein data")
    parser.add_argument("--data_file", type=str, required=True, help="Path to train CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed files")
    args = parser.parse_args()

    preprocess_data(args.data_file, args.output_dir)
