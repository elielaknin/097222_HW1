import pandas as pd
import os
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from statistics import mode


def get_video_stats(predicted_gt_frames_df, stats_folder_path):
    os.makedirs(stats_folder_path, exist_ok=True)

    GT = list(predicted_gt_frames_df['right_gt']) + list(predicted_gt_frames_df['left_gt'])
    predicted = list(predicted_gt_frames_df['right_predict']) + list(predicted_gt_frames_df['left_predict'])
    labels = ['Right_Scissors', 'Left_Scissors', 'Right_Needle_driver', 'Left_Needle_driver', 'Right_Forceps',
              'Left_Forceps', 'Right_Empty', 'Left_Empty']

    class_report = metrics.classification_report(GT, predicted, labels=labels, digits=3)
    overall_accuracy = round(metrics.accuracy_score(GT, predicted), 3)

    # open file
    path_of_file = os.path.join(stats_folder_path, 'metrics_results.txt')
    text_file = open(path_of_file, "w")
    # write string to file
    text_file.write(class_report)
    text_file.write(f"\n overall accuracy: {overall_accuracy}")
    # close file
    text_file.close()

    #save confussion metrics
    cm = metrics.confusion_matrix(GT, predicted, labels=labels)
    pd.DataFrame(cm).to_csv(os.path.join(stats_folder_path, 'cm.csv'))
    cmp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(16, 16))
    cmp.plot(ax=ax)
    plt.xticks(rotation=45, ha="right")
    plt.savefig(os.path.join(stats_folder_path, 'cm.png'))


def get_video_time_label(video_name, video_time_label_path):
    dict_tool_usage = {'T0': 'Empty', 'T1': 'Needle_driver', 'T2': 'Forceps', 'T3': 'Scissors'}

    dict_hand_label = {}
    #get left tool usage of the video
    for hand in ['Left', 'Right']:
        if hand == 'Left':
            tool_path = 'tools_left'
        else:
            tool_path = 'tools_right'

        hand_path = os.path.join(video_time_label_path, tool_path, f"{video_name}.txt")
        hand_gt_label = pd.read_csv(hand_path, sep=" ", header=None)
        hand_gt_label.rename(columns={0: 'start_time', 1: 'end_time', 2: 'label'}, inplace=True)

        name_list = []
        for idx, row in hand_gt_label.iterrows():
            label_t = row['label']
            name = hand + "_" + dict_tool_usage[label_t]
            name_list.append(name)
        hand_gt_label['name'] = name_list

        dict_hand_label[hand] = hand_gt_label
    return dict_hand_label


def get_gt_label(full_hand_label_dict, frameNr):
    right_df = full_hand_label_dict['Right']
    for idx, row in right_df.iterrows():
        if frameNr >= row['start_time'] and frameNr <= row['end_time']:
            right_name_gt = row['name']
            break

    left_df = full_hand_label_dict['Left']
    for idx, row in left_df.iterrows():
        if frameNr >= row['start_time'] and frameNr <= row['end_time']:
            left_name_gt = row['name']
            break

    return right_name_gt, left_name_gt


def smooth_confidence_function(tools_df, history_label, history_bbox):
    # use history previous bbox and label if one condition exist:
    #   - detection df is empty
    #   - detection df has more than one object detection
    #   - detection confidence is less than 75

    if len(tools_df) != 1:
        bbox = history_bbox[-1:][0]
        label = history_label[-1:][0]
        return bbox, label

    tools_sorted_df = tools_df.sort_values(by=['confidence'], ascending=False)
    if tools_sorted_df['confidence'].iloc[0] < 0.75:
        # take info from previous frame
        bbox = history_bbox[-1:][0]
        label = history_label[-1:][0]
    else:
        label = tools_sorted_df['name'].iloc[0]
        bbox = list(tools_sorted_df[['xmin', 'ymin', 'xmax', 'ymax']].astype('int').iloc[0])

    return bbox, label


def smooth_function(tools_df, history_queue):
    #check number of labels in df
    number_of_labels = len(np.unique(tools_df['name']))

    if number_of_labels == 1:
        # only one label exist so just average bounding box and return
        label = np.unique(tools_df['name'])[0]
        bbox = list(np.mean(tools_df[['xmin', 'ymin', 'xmax', 'ymax']], axis=0).astype('int'))

    else:
        # check what is the most common class from the last 30 frames
        if len(history_queue) >= 10:
            history_label = mode(history_queue[-10:])
        else:
            history_label = mode(history_queue)

        # check if history most common label exist in df
        # if yes take it else take the label with the higher confidence
        if history_label in tools_df['name'].values:
            label = history_label
            tools_new_label_df = tools_df[tools_df['name'] == label]
            bbox = list(np.mean(tools_new_label_df[['xmin', 'ymin', 'xmax', 'ymax']], axis=0).astype('int'))
        else:
            tools_new_label_df = tools_df.sort_values(by=['confidence'], ascending=False)
            label = tools_new_label_df['name'].iloc[0]
            bbox = list(tools_new_label_df[['xmin', 'ymin', 'xmax', 'ymax']].astype('int').iloc[0])

    return bbox, label
