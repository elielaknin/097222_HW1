import pandas as pd
import os
import argparse
import torch


def predict_function(model_path, image_path):

    # Model
    model = torch.hub.load('yolov5/', 'custom', path=model_path, source='local') # or yolov5n - yolov5x6, custom

    # Inference on one image
    results = model(image_path)

    return results.pandas().xywh[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument('--image_path', type=str, default='', help='image path for inference')

    args = parser.parse_args()


    predict_function(args.model_path, args.images_folder)


if __name__ == '__main__':
    main()