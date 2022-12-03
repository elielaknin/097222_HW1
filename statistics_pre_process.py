import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt


def get_statistics_bboxes():
    bboxes_labels_folder_path = 'data/images/bboxes_labels'
    classes_names_file_path = 'data/images/classes.names'

    classes_names_dict = get_classes_dict(classes_names_file_path)

    histogram_arr = np.zeros(8)
    for txt_file in os.listdir(bboxes_labels_folder_path):
        with open(os.path.join(bboxes_labels_folder_path, txt_file)) as f:
            for line in f:
                class_number = line.split(' ')[0]
                histogram_arr[int(class_number)] += 1
        f.close()

    plt.figure(figsize=(12, 12))
    plt.bar(list(classes_names_dict.values()), histogram_arr, color=['cadetblue'])
    plt.xticks(rotation=45)
    plt.ylabel('Frequency appearance of data', fontsize=12)
    plt.title('Histogram of bounding box labels', fontsize=18)
    plt.show()
    print('a')


def get_classes_dict(classes_names_file_path):
    classes_names_dict = {}
    with open(classes_names_file_path) as f:
        idx = 0
        for line in f:
            classes_names_dict[idx] = line.split('\n')[0]
            idx += 1
    f.close()

    return classes_names_dict


def main():
    get_statistics_bboxes()


if __name__ == '__main__':
    main()