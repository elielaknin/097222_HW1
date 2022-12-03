import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import shutil


def split_images_dataset():
    dataset_folder_path = 'data/images/'
    output_dataset_path = 'dataset/images'


    for txt_file, dst_folder_name in [('train.txt', 'train'), ('valid.txt', 'val'), ('test.txt', 'test')]:
        dst_folder = os.path.join(output_dataset_path, dst_folder_name)
        os.makedirs(dst_folder, exist_ok=True)
        with open(os.path.join(dataset_folder_path, txt_file)) as f:
            for image_name in f:
                src_image = os.path.join(dataset_folder_path, 'images', image_name.split('\n')[0])
                shutil.copy2(src_image, dst_folder)
        f.close()


def split_labels_dataset():
    dataset_folder_path = 'data/images/'
    output_dataset_path = 'dataset/labels'

    for txt_file, dst_folder_name in [('train.txt', 'train'), ('valid.txt', 'val'), ('test.txt', 'test')]:
        dst_folder = os.path.join(output_dataset_path, dst_folder_name)
        os.makedirs(dst_folder, exist_ok=True)
        with open(os.path.join(dataset_folder_path, txt_file)) as f:
            for image_name in f:
                src_label = os.path.join(dataset_folder_path, 'bboxes_labels', image_name.split('.jpg')[0])
                src_label += '.txt'
                shutil.copy2(src_label, dst_folder)
        f.close()



def main():
    split_labels_dataset()
    # split_images_dataset()


if __name__ == '__main__':
    main()