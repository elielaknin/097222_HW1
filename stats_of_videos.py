import pandas as pd
import argparse
from utils_functions import get_video_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_csv_path', type=str, help='model path')
    parser.add_argument('--output_folder', type=str, help='model path')
    args = parser.parse_args()

    df = pd.read_csv(args.video_csv_path, index_col='frame_number')

    get_video_stats(df, args.output_folder)


if __name__ == '__main__':
    main()