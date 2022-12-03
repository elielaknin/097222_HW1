import pandas as pd
import os


def train_yolo():

    # Train YOLOv5s
    yaml_file = 'yolov5/hw1_yolov5.yaml'
    img_size = 640
    epochs_number = 200
    # batch_size = 16
    # pretrained_weights = 'yolov5s.pt'

    batch_size_list = [4, 8, 16, 32, 64, 128]
    optimizer_list = ['SGD', 'Adam', 'AdamW']
    pretrained_weights_list = ['', 'yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt']


    for batch_size in batch_size_list:
        for optimizer in optimizer_list:
            for pretrained_weights in pretrained_weights_list:
                os.system(f"python yolov5/train.py --img {img_size} --batch {batch_size} --epochs {epochs_number} "
                          f"--data {yaml_file} --weights {pretrained_weights} --optimizer {optimizer}")


def main():
    train_yolo()

if __name__ == '__main__':
    main()