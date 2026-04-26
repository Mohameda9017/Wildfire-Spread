from pathlib import Path
import torch
import tensorflow as tf
import numpy as np
from tqdm import tqdm

INPUT_FEATURES = ['elevation', 'th', 'vs', 'tmmn', 'tmmx',
    'sph', 'pr', 'pdsi', 'NDVI', 'population',
    'erc', 'PrevFireMask']

OUTPUT_FEATURES = ['FireMask']

def get_features_dict(sample_size: int =64) -> dict:
    '''
    A Tfrecord is serialized meanign tensorflow doesnt know what features are inside, the shape of it, and and data types each one has
    So we need to tell it what features to expect and what data type they are. 

    Creates the dictionary that tells TensorFlow how to read each field inside one TFRecord example.
    Each feature in this dataset is a 64x64 float grid.
    '''

    feature_dict = {}
    for key in INPUT_FEATURES + OUTPUT_FEATURES:
        # for each feature, the value inside the TFRecord is a fixed-size tensor ( same shape, same data type) of shape (64*64)
        feature_dict[key] = tf.io.FixedLenFeature(shape=[sample_size, sample_size],dtype=tf.float32) 
    return feature_dict

def parse_tfrecord(tfrecord : tf.Tensor, sample_size: int = 64):
    """ Parses a Single TFrecord and returns:
        input_img: numpy array of shape [H, W, C]
        label_img: numpy array of shape [H, W, 1]
    """
    features = tf.io.parse_single_example(tfrecord, features=get_features_dict(sample_size)) # Take this raw encoded sample and decode it using the dict schema. 
    # features["elevation"]   # this would give me the acuatl 64*64 tensor for elevation.

    input_list = [] # holds all the input features as tensors
    for key in INPUT_FEATURES:
        input_list.append(features[key]) # this is a list of tensors, each of shape (64, 64)
    inputs_stacked = tf.stack(input_list, axis=0) # combines the 12 tensors into one tensor of 12,64,64
    input_image = tf.transpose(inputs_stacked, [1,2,0]) # rearranges the dimensions to be 64,64,12.

    label_image = features["FireMask"] 
    label_img = tf.expand_dims(label_image, axis=-1)

    return input_image.numpy(), label_img.numpy()

def process_sample(input_img : np.ndarray, label_img : np.ndarray):
    '''
    Takes in a parsed tf record sample and converts it into the format we want for our PyTorch model.
    Input:
        input_img  -> [H, W, C]
        label_img  -> [H, W, 1]

    Output:
        image      -> [C, H, W]
        label      -> [1, H, W]
        valid_mask -> [1, H, W]

    Note:
        - In the firemask, there are unlabled pixels with value -1. 
    '''

    # Checks every pixel, and if its -1 then its 0, otherwise its 1. Will use later for loss calculation to ignore the unlabeled pixels.
    valid_mask = (label_img != -1).astype(np.float32) 

    binary_label = np.where(label_img == 1, 1.0, 0.0).astype(np.float32) # convert the firemask into binary labels where 1 is fire and 0 is no fire.

    image = np.transpose(input_img, (2, 0, 1)).astype(np.float32) # rearrange dimensions to be C,H,W for pytorch
    label = np.transpose(binary_label, (2, 0, 1)).astype(np.float32)
    valid_mask = np.transpose(valid_mask, (2, 0, 1)).astype(np.float32)

    return image, label, valid_mask

def convert_split(input_pattern: Path, output_dir: Path, sample_size: int = 64):
    """
    Converts all TFrecord files from one split (train/val/test) into PyTorch format and saves each one as .pt file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tfrecord_files = sorted(input_pattern.parent.glob(input_pattern.name)) # gets a list of all the tfrecord files that match the pattern.

    print(f"Found {len(tfrecord_files)} TFRecord files for pattern: {input_pattern}")

    if len(tfrecord_files) == 0:
        print("No TFRecord files found")
        return 
    
    sample_id = 0
    for tf_recordfile in tfrecord_files:
        dataset = tf.data.TFRecordDataset(str(tf_recordfile)) # creates a dataset object that can read the tfrecord file.
        # loop through each sample in the tfrecord file
        for raw_record in tqdm(dataset, desc=f"Converting {tf_recordfile.name}"):
            input_img, label_img = parse_tfrecord(raw_record, sample_size=sample_size) # parse the raw record into input and label images
            image, label, valid_mask = process_sample(input_img, label_img)

            sample = {
            "image": torch.from_numpy(image), # convert the numpy arrays into pytorch tensors
            "label": torch.from_numpy(label),
            "valid_mask": torch.from_numpy(valid_mask),
            }

            save_path = output_dir / f"sample_{sample_id}.pt"

            torch.save(sample, save_path)
            sample_id += 1
    
    print(f"Finished. Saved {sample_id} samples to {output_dir}")


def main():
    project_root = Path(__file__).resolve().parents[2]

    split_patterns = {
        "train": project_root / "data" / "raw_tfrecords" / "next_day_wildfire_spread_train*.tfrecord",
        "eval":  project_root / "data" / "raw_tfrecords" / "next_day_wildfire_spread_eval*.tfrecord",
        "test":  project_root / "data" / "raw_tfrecords" / "next_day_wildfire_spread_test*.tfrecord",
    }

    for split_name, pattern_path in split_patterns.items():
        output_dir = project_root / "data" / "processed_pt" / split_name
        print(f"\n--- Converting {split_name} split ---")
        convert_split(pattern_path, output_dir, sample_size=64)


if __name__ == "__main__":
    main()










