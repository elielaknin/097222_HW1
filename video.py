import pandas as pd
import os
import argparse
import torch
import cv2
import bbox_visualizer as bbv
from tqdm import tqdm

from utils_functions import get_video_stats, get_video_time_label ,get_gt_label, smooth_function, \
    smooth_confidence_function

BLUE_COLOR = (225, 224, 145)
RED_COLOR = (60, 20, 220)

"""Predict video with bounding box and class.

Using video and video time label. Split the video to frames and predict each frame
Using yolo v5 model path. Then create new frame with GT and predicited bbox and label.
Create new video based on annoted frames and return statistic of the video (CM, recall, precision, f1-score).

Parameters
----------
model_path : yolo v5 ".pt" file path
video_path : path of the video
video_label_path: location of the video label (tool_usage)

Return
------
None

"""
def video(model_path, video_path, video_label_path):

    # Load model
    model = torch.hub.load('yolov5/', 'custom', path=model_path, source='local') # or yolov5n - yolov5x6, custom

    #get video path
    video_name = os.path.basename(video_path).split('.wmv')[0]
    video_folder_path = os.path.dirname(video_path)

    general_output_folder = os.path.join(video_folder_path, video_name)
    os.makedirs(general_output_folder, exist_ok=True)

    new_predicted_video_path = os.path.join(general_output_folder, (video_name + '_predicted.avi'))

    # if label exist get usage tool for right and left
    if video_label_path:
        predicted_gt_frames_dict = {}# right_gt, right_predict, left_gt, left_predict
        full_hand_label_dict = get_video_time_label(video_name, video_label_path)

    #Split video to frames
    # read video
    capture = cv2.VideoCapture(video_path)
    number_of_frames_in_video = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # save new video
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    out = cv2.VideoWriter(new_predicted_video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    # Check if camera opened successfully
    if (capture.isOpened() == False):
        print("Error opening video stream or file")

    history_label_left, history_bbox_left = [], []
    history_label_right, history_bbox_right = [], []

    for frameNr in tqdm(range(number_of_frames_in_video)):
    # for frameNr in tqdm(range(500)):

        success, frame = capture.read()
        if success:

            #save frame
            cv2.imwrite('tmp_image.jpg', frame)
            predicted_df = model('tmp_image.jpg').pandas().xyxy[0]
            predicted_frame = frame

            left_tools_df = predicted_df[predicted_df['class'] % 2 == 1]

            # option B
            left_bbox, left_label = smooth_confidence_function(left_tools_df, history_label_left, history_bbox_left)

            # if not left_tools_df.empty:
                # if len(left_tools_df) > 1:
                #     # if prediction of the tool is not sure (2 or more) use history to decide
                #     # also if the modl detect 2 or more bbox for the tool, then average the bbox points
                #     left_bbox, left_label = smooth_function(left_tools_df, history_queue_left)
                # else:
                #     left_bbox = list(left_tools_df[['xmin', 'ymin', 'xmax', 'ymax']].astype('int').iloc[0])
                #     left_label = left_tools_df['name'].iloc[0]

            history_label_left.append(left_label)
            history_bbox_left.append(left_bbox)

            predicted_frame = bbv.draw_rectangle(predicted_frame, left_bbox, bbox_color=BLUE_COLOR)
            predicted_frame = bbv.add_label(predicted_frame, left_label, left_bbox, top=True, text_bg_color=BLUE_COLOR)


            right_tools_df = predicted_df[predicted_df['class'] % 2 == 0]

            # option B
            right_bbox, right_label = smooth_confidence_function(right_tools_df, history_label_right, history_bbox_right)

            # if not right_tools_df.empty:
                # if len(right_tools_df) > 1:
                #     # if prediction of the tool is not sure (2 or more) use history to decide
                #     # also if the modl detect 2 or more bbox for the tool, then average the bbox points
                #     right_bbox, right_label = smooth_function(right_tools_df, history_queue_right)
                # else:
                #     right_bbox = list(right_tools_df[['xmin', 'ymin', 'xmax', 'ymax']].astype('int').iloc[0])
                #     right_label = right_tools_df['name'].iloc[0]

            history_label_right.append(right_label)
            history_bbox_right.append(right_bbox)

            predicted_frame = bbv.draw_rectangle(predicted_frame, right_bbox, bbox_color=RED_COLOR)
            predicted_frame = bbv.add_label(predicted_frame, right_label, right_bbox, top=True, text_bg_color=RED_COLOR)

            # if label exist calculat stats
            if video_label_path:
                right_gt, left_gt = get_gt_label(full_hand_label_dict, frameNr)
                predicted_gt_frames_dict[frameNr] = (right_gt, right_label, left_gt, left_label)

                #add GT label on the image
                predicted_frame = cv2.putText(predicted_frame, "GT: " + right_gt, (0, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                              1, RED_COLOR, 2, 2)
                predicted_frame = cv2.putText(predicted_frame, "GT: " + left_gt, (0, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                              1, BLUE_COLOR, 2, 2)

            out.write(predicted_frame)
        else:
            break

    os.remove('tmp_image.jpg')
    capture.release()
    out.release()


    predicted_gt_frames_df = pd.DataFrame(predicted_gt_frames_dict, index=['right_gt', 'right_predict', 'left_gt',
                                                                           'left_predict']).T
    predicted_gt_frames_df.index.name = 'frame_number'

    video_stats_folder = os.path.join(general_output_folder, 'video_stats')
    get_video_stats(predicted_gt_frames_df, video_stats_folder)

    frames_stats_path = os.path.join(video_stats_folder, 'stats_by_frames.csv')
    predicted_gt_frames_df.to_csv(frames_stats_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument('--video_path', type=str, help='model path')
    parser.add_argument('--video_label_path', nargs='?', help='Add GT time label folder to get video stats')

    args = parser.parse_args()

    video(args.model_path, args.video_path, args.video_label_path)

if __name__ == '__main__':
    main()