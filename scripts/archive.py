import json
import dicttoxml
import os
import re
from collections import Counter
from datetime import date
import glob
import cv2


def read_json(path):
    with open(path, 'r') as f:
        out_str = json.load(f)
        return out_str


def write_file(tool_name, out_str, out_path, make_json=True, make_xml=True):
    today = date.today()
    d1 = today.strftime("%d.%m.%Y")

    if make_json:
        with open(out_path + '/' + tool_name + '_' + d1 + '.json', 'w') as json_file:
            json.dump(out_str, json_file)

    if make_xml:
        with open(out_path + '/' + tool_name + '_' + d1 + '.xml', 'w') as xml_file:
            xml_file.write(dicttoxml.dicttoxml(out_str).decode())


def save_predictions(root_path, predictor):
    root_json_path = root_path + '/json/*'

    folders = [f for f in glob.glob(root_json_path, recursive=True)]

    out_str = Counter()

    for folder in folders:
        generate_structure(folder, out_str, predictor)

    write_file(predictor, out_str, root_path)


def load_metadata(video, frame, predictor, frame_time=0, video_folder=None):
    if predictor == 'wrnchai':
        video['people_no'] = len(frame['persons'])
        video['width'] = frame['width']
        video['height'] = frame['height']

    elif predictor == 'openpose' or predictor == 'dlib' or predictor == 'mtcnn':
        if predictor == 'openpose' or predictor == 'dlib':
            video['people_no'] = len(frame['people'])
        elif predictor == 'mtcnn':
            video['people_no'] = len(frame)

        video_file = [f for f in glob.glob(video_folder, recursive=True)][0]
        cvObj = cv2.VideoCapture(video_file)

        if cvObj.isOpened():
            cvObj.set(cv2.CAP_PROP_POS_FRAMES, frame_time)
            video['width'] = int(cvObj.get(cv2.CAP_PROP_FRAME_WIDTH))
            video['height'] = int(cvObj.get(cv2.CAP_PROP_FRAME_HEIGHT))


def load_info(pose_in, pose_out, dimension, limb, *keypoints):
    pose_out[limb] = list()
    ranges = keypoints[0]
    points = keypoints[1]
    vector_point = list()

    if ranges is not None:
        vector_point = [x for i, x in enumerate(pose_in) for range in ranges if
                        range[0] * dimension <= i < (range[1] + 1) * dimension]

    if points is not None:
        vector_point.extend([x for i, x in enumerate(pose_in) for point in points if
                             point * dimension <= i < (point + 1) * dimension])

    if ranges is None and points is None:
        vector_point = [x for x in pose_in]

    temp = list()
    for i, point in enumerate(vector_point):
        if type(point) == type(list()):
            temp.extend(point)
        else:
            temp.append(point)

        if dimension == 1:
            pose_out[limb].append(temp)
            temp = list()
        elif (i+1) % dimension == 0:
            pose_out[limb].append(tuple(temp))
            temp = list()


def wrnchai_data(data_in, data_out, data_type, human_set, dimension):
    if human_set not in data_out.keys():
        data_out[human_set] = Counter()

    if human_set != 'hand':
        data_out[human_set][dimension] = Counter()
    elif 'left' not in data_out[human_set].keys() or 'left' not in data_out[human_set].keys():
            data_out[human_set]['left'] = Counter()
            data_out[human_set]['right'] = Counter()

    data = list()

    if dimension == '2d':
        if human_set+'2d' in data_in.keys() and human_set == 'pose':
            if data_type == 'keypoints':
                load_info(data_in['pose2d']['joints'], data_out[human_set][dimension], 2, 'head', [(16, 20)], (8, 9))
                load_info(data_in['pose2d']['joints'], data_out[human_set][dimension], 2, 'spine', None, (6, 7))
                load_info(data_in['pose2d']['joints'], data_out[human_set][dimension], 2, 'left_arm', [(10, 12)], None)
                load_info(data_in['pose2d']['joints'], data_out[human_set][dimension], 2, 'right_arm', [(13, 15)], None)
                load_info(data_in['pose2d']['joints'], data_out[human_set][dimension], 2, 'left_leg', [(3, 5)], (22, 24))
                load_info(data_in['pose2d']['joints'], data_out[human_set][dimension], 2, 'right_leg', [(0, 2)], (21, 23))

            elif data_type == 'confidence':
                load_info(data_in['pose2d']['scores'], data_out[human_set][dimension], 1, 'head', [(16, 20)], (8, 9))
                load_info(data_in['pose2d']['scores'], data_out[human_set][dimension], 1, 'spine', None, (6, 7))
                load_info(data_in['pose2d']['scores'], data_out[human_set][dimension], 1, 'left_arm', [(10, 12)], None)
                load_info(data_in['pose2d']['scores'], data_out[human_set][dimension], 1, 'right_arm', [(13, 15)], None)
                load_info(data_in['pose2d']['scores'], data_out[human_set][dimension], 1, 'left_leg', [(3, 5)], (22, 24))
                load_info(data_in['pose2d']['scores'], data_out[human_set][dimension], 1, 'right_leg', [(0, 2)], (21, 23))

            elif data_type == 'box':
                data.extend([data_in['pose2d']['bbox']['minX'], data_in['pose2d']['bbox']['minY'],
                             data_in['pose2d']['bbox']['width'], data_in['pose2d']['bbox']['height']])

                load_info(data, data_out[human_set][dimension], 2, 'point', None, (0,))
                load_info(data, data_out[human_set][dimension], 2, 'size', None, (1,))

        elif human_set+'_pose' in data_in.keys() and human_set == 'hand':
            if 'left' in data_in['hand_pose'].keys():
                if data_type == 'keypoints':
                    load_info(data_in['hand_pose']['left']['joints'], data_out[human_set]['left'], 2, dimension, None, None)
                elif data_type == 'confidence':
                    load_info(data_in['hand_pose']['left']['scores'], data_out[human_set]['left'], 1, dimension, None, None)
                elif data_type == 'box':
                    data.extend([data_in['hand_pose']['left']['bbox']['minX'], data_in['hand_pose']['left']['bbox']['minY'],
                                 data_in['hand_pose']['left']['bbox']['width'], data_in['hand_pose']['left']['bbox']['height']])
                    data_out[human_set]['left'][dimension] = Counter()

                    load_info(data, data_out[human_set]['left'][dimension], 2, 'point', None, (0,))
                    load_info(data, data_out[human_set]['left'][dimension], 2, 'size', None, (1,))

            elif 'right' in data_in['hand_pose'].keys():
                if data_type == 'keypoints':
                    load_info(data_in['hand_pose']['right']['joints'], data_out[human_set]['right'], 2, dimension, None, None)
                elif data_type == 'confidence':
                    load_info(data_in['hand_pose']['right']['scores'], data_out[human_set]['right'], 1, dimension, None, None)
                elif data_type == 'box':
                    data.extend(
                        [data_in['hand_pose']['right']['bbox']['minX'], data_in['hand_pose']['right']['bbox']['minY'],
                         data_in['hand_pose']['right']['bbox']['width'], data_in['hand_pose']['right']['bbox']['height']])
                    data_out[human_set]['right'][dimension] = Counter()

                    load_info(data, data_out[human_set]['right'][dimension], 2, 'point', None, (0,))
                    load_info(data, data_out[human_set]['right'][dimension], 2, 'size', None, (1,))

    elif dimension == '3d':
        if human_set+'_3d_raw' in data_in.keys() and human_set == 'pose':
            if data_type == 'keypoints':
                load_info(data_in['pose_3d_raw']['positions'], data_out[human_set][dimension], 3, 'head', [(16, 20)], (8, 9))
                load_info(data_in['pose_3d_raw']['positions'], data_out[human_set][dimension], 3, 'spine', None, (6, 7))
                load_info(data_in['pose_3d_raw']['positions'], data_out[human_set][dimension], 3, 'left_arm', [(10, 12)], None)
                load_info(data_in['pose_3d_raw']['positions'], data_out[human_set][dimension], 3, 'right_arm', [(13, 15)], None)
                load_info(data_in['pose_3d_raw']['positions'], data_out[human_set][dimension], 3, 'left_leg', [(3, 5)], (22, 24))
                load_info(data_in['pose_3d_raw']['positions'], data_out[human_set][dimension], 3, 'right_leg', [(0, 2)], (21, 23))


def openpose_data(data_in, data_out, data_type, human_set, dimension):
    if human_set not in data_out.keys():
        data_out[human_set] = Counter()

    if human_set != 'hand':
        data_out[human_set][dimension] = Counter()

    keypoints = list()
    scores = list()
    data = None
    dim = 0

    if human_set + '_keypoints_' + dimension in data_in.keys() and human_set in ['pose', 'face']:
        for i, x in enumerate(data_in[human_set +'_keypoints_' + dimension]):
            if (i+1) % (int(dimension[0]) + 1) == 0:
                scores.append(x)
            else:
                keypoints.append(x)

    if data_type == 'keypoints':
        data = keypoints
        dim = int(dimension[0])
    elif data_type == 'confidence':
        data = scores
        dim = 1

    if human_set == 'pose':
        if len(keypoints) == 25*2:
            load_info(data, data_out[human_set][dimension], dim, 'head', [(15, 18)], (0,))
            load_info(data, data_out[human_set][dimension], dim, 'spine', None, (1, 8))
            load_info(data, data_out[human_set][dimension], dim, 'right_arm', [(2, 4)], None)
            load_info(data, data_out[human_set][dimension], dim, 'left_arm', [(5, 7)], None)
            load_info(data, data_out[human_set][dimension], dim, 'right_leg', [(9, 11), (22, 24)], None)
            load_info(data, data_out[human_set][dimension], dim, 'left_leg', [(12, 14), (19, 21)], None)

        elif len(keypoints) == 18*2:
            load_info(data, data_out[human_set][dimension], dim, 'head', [(14, 17)], (0))
            load_info(data, data_out[human_set][dimension], dim, 'spine', None, (1))
            load_info(data, data_out[human_set][dimension], dim, 'right_arm', [(2, 4)], None)
            load_info(data, data_out[human_set][dimension], dim, 'left_arm', [(5, 6)], None)
            load_info(data, data_out[human_set][dimension], dim, 'right_leg', [(8, 10)], None)
            load_info(data, data_out[human_set][dimension], dim, 'left_leg', [(11, 13)], None)

    elif human_set == 'face':
        load_info(data, data_out[human_set][dimension], dim, 'jaw', [(0, 16)], None)
        load_info(data, data_out[human_set][dimension], dim, 'left_eye', [(22, 26), (42,47)], (69,))
        load_info(data, data_out[human_set][dimension], dim, 'right_eye', [(17 ,21), (36,41)], (68,))
        load_info(data, data_out[human_set][dimension], dim, 'nose', [(27, 35)], None)
        load_info(data, data_out[human_set][dimension], dim, 'mouth', [(48, 67)], None)

    elif human_set == 'hand':
        for side in ['left', 'right']:
            if human_set + '_' + side + '_keypoints_' + dimension in data_in.keys():
                for i, x in enumerate(data_in[human_set + '_' + side + '_keypoints_' + dimension]):
                    if (i + 1) % (int(dimension[0]) + 1) == 0:
                        scores.append(x)
                    else:
                        keypoints.append(x)

            if side not in data_out[human_set].keys():
                data_out[human_set][side] = Counter()

            load_info(data, data_out[human_set][side], dim, dimension, [(1,4)], (0,))


def dlib_data(data_in, data_out, data_type, human_set, dimension):
    if human_set not in data_out.keys():
        data_out[human_set] = Counter()

    data_out[human_set][dimension] = Counter()

    if human_set + '_' + data_type + '_' + dimension not in data_in:
        return

    data = data_in[human_set+'_' + data_type + '_' + dimension]
    dim = 0

    if data_type == 'keypoints':
        dim = int(dimension[0])
    elif data_type == 'confidence':
        dim = 1

    if human_set == 'face':
        load_info(data, data_out[human_set][dimension], dim, 'left_eye', [(22, 26), (42, 47)], None)
        load_info(data, data_out[human_set][dimension], dim, 'right_eye', [(17, 21), (36, 41)], None)
        load_info(data, data_out[human_set][dimension], dim, 'nose', [(27, 35)], None)
        load_info(data, data_out[human_set][dimension], dim, 'mouth', [(48, 67)], None)
        load_info(data, data_out[human_set][dimension], dim, 'jaw', [(0,16)], None)


def mtcnn_data(data_in, data_out, data_type, human_set, dimension):
    if human_set not in data_out.keys():
        data_out[human_set] = Counter()

    data_out[human_set][dimension] = Counter()

    if human_set != 'face':
        return

    data = list()
    if data_type == 'keypoints':
        data.extend(data_in[data_type]['left_eye'])
        data.extend(data_in[data_type]['right_eye'])
        data.extend(data_in[data_type]['mouth_left'])
        data.extend(data_in[data_type]['mouth_right'])
        data.extend(data_in[data_type]['nose'])
    elif data_type == 'confidence':
        data.append(data_in[data_type])
    elif data_type == 'box':
        data = list()
        data.extend([data_in['box'][0], data_in['box'][1], data_in['box'][2], data_in['box'][3]])

    dim = 0

    if data_type == 'keypoints':
        dim = int(dimension[0])
    elif data_type == 'confidence':
        dim = 1

    if dimension == '2d':
        if data_type == 'keypoints':
            load_info(data, data_out[human_set][dimension], dim, 'left_eye', None, (0,))
            load_info(data, data_out[human_set][dimension], dim, 'left_eye', None, (1,))
            load_info(data, data_out[human_set][dimension], dim, 'mouth', [(2, 3)], None)
            load_info(data, data_out[human_set][dimension], dim, 'nose', None, (4,))
            load_info(data, data_out[human_set][dimension], dim, 'box', None, (5,))
        elif data_type == 'confidence':
            load_info(data, data_out[human_set], dim, '2d', None, (0,))
        elif data_type == 'box':
            load_info(data, data_out[human_set][dimension], 2, 'point', None, (0,))
            load_info(data, data_out[human_set][dimension], 2, 'size', None, (1,))


def data_generator(data_in, data_out, data_type, predictor):
    for human_set in ['pose', 'face', 'hand']:
        for dimension in ['2d', '3d']:
            if predictor == 'wrnchai':
                wrnchai_data(data_in, data_out, data_type, human_set, dimension)
            elif predictor == 'openpose':
                openpose_data(data_in, data_out, data_type, human_set, dimension)
            elif predictor == 'dlib':
                dlib_data(data_in, data_out, data_type, human_set, dimension)
            elif predictor == 'mtcnn':
                mtcnn_data(data_in, data_out, data_type, human_set, dimension)


def load_data(data_out, data_in, data_type, predictor):
    if 'people' not in data_out.keys():
        data_out['people'] = list()

    persons = list()

    if predictor == 'wrnchai':
        persons = data_in['persons']
    elif predictor == 'openpose' or predictor == 'dlib':
        persons = data_in['people']
    elif predictor == 'mtcnn':
        persons = data_in

    for id, person in enumerate(persons):
        if len(data_out['people']) < data_out['people_no']:
            data_out['people'].append(Counter())

        data_out['people'][id][data_type] = Counter()
        data_generator(person, data_out['people'][id][data_type], data_type, predictor)


def generate_structure(folder, out_dict, predictor):
    out_dict[predictor] = Counter()
    video = os.path.basename(folder)
    out_dict[predictor][video] = Counter()

    json_path = folder + '/*.json'
    files = [f for f in glob.glob(json_path, recursive=True)]
    files.sort()

    if predictor == 'wrnchai':
        file = files[0]
        in_dict = read_json(file)

        for i, frame in enumerate(in_dict['frames']):
            out_dict[predictor][video][frame['frame_time']] = Counter()
            load_metadata(out_dict[predictor][video][frame['frame_time']], frame, predictor)
            load_data(out_dict[predictor][video][frame['frame_time']], frame, 'keypoints', predictor)
            load_data(out_dict[predictor][video][frame['frame_time']], frame, 'confidence', predictor)
            load_data(out_dict[predictor][video][frame['frame_time']], frame, 'box', predictor)

    else:
        for file in files:
            p = re.search(r"\d{9}", file)
            frame_time = int(p.group())
            out_dict[predictor][video][frame_time] = Counter()

            frame = read_json(file)

            load_metadata(out_dict[predictor][video][frame_time], frame, predictor, frame_time, 'videos/' + video + '*')
            load_data(out_dict[predictor][video][frame_time], frame, 'keypoints', predictor)
            load_data(out_dict[predictor][video][frame_time], frame, 'confidence', predictor)
            load_data(out_dict[predictor][video][frame_time], frame, 'box', predictor)
            pass


# save_predictions('out/wrnchAI_20.09.2019', 'wrnchai') boxed
# save_predictions('out/openpose_15.09.2019', 'openpose') no boxed
# save_predictions('out/dlib_16.09.2019', 'dlib') no boxed, no confidence
# save_predictions('out/mtcnn_17.09.2019', 'mtcnn') boxed