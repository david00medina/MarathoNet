import os
import cv2 as cv
from datetime import date
from openpose import pyopenpose as op


def read_video_files(path):
    video_files = list()
    for root, dirs, files in os.walk(os.path.relpath(path)):
        for file in files:
            video_files.append(os.path.join(root, file))

    return video_files


def generate_video_capture(video_files):
    keys = [ 'video_' + str(i+1) for i in range(len(video_files)) ]

    videos = { k: cv.VideoCapture(cv.samples.findFileOrKeep(v)) for (k, v) in zip(keys, video_files) }

    for k,v in videos.items():
        if not v.isOpened():
            print('Unable to open: \'' + k + '\'')
            exit(0)

    return videos


def openpose_setup(json_path='', model_folder='models/', render_pose=2,
                   face=True, face_detector=1, face_render=-1,
                   hand=True, hand_detector=3, hand_render=-1):
    params = dict()
    params['write_json'] = json_path
    params['model_folder'] = model_folder
    params['render_pose'] = render_pose
    params['body'] = 1
    # params['face'] = face
    # params['face_detector'] = face_detector
    # params['face_render'] = face_render
    # params['hand'] = hand
    # params['hand_detector'] = hand_detector
    # params['hand_render'] = hand_render

    return params


def define_size(inSize, out_size=(800, 600)):
    w, h = inSize

    if w > out_size[0]:
        scaling = out_size[0] / w
        w = int(w * scaling)
        h = int(h * scaling)

    if h > out_size[1]:
        scaling = out_size[1] / h
        w = int(w * scaling)
        h = int(h * scaling)

    return w, h


def resize_frame(frame, inSize, outSize=(800, 600)):
    dim = define_size(inSize, outSize)

    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)


def infere(predictor, i, file_i, out_path, videos, size=(800, 600)):
    datum = None
    handler = None

    if predictor == 'openpose':
        handler = op.WrapperPython()
        params = openpose_setup(out_path + "json/" + file_i)
        handler.configure(params)
        handler.start()
        datum = op.Datum()

    total_frames = int(videos['video_' + str(i + 1)].get(cv.CAP_PROP_FRAME_COUNT))
    w = int(videos['video_' + str(i + 1)].get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(videos['video_' + str(i + 1)].get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(videos['video_' + str(i + 1)].get(cv.CAP_PROP_FPS))

    videos['video_' + str(i + 1)].set(cv.CAP_PROP_POS_FRAMES, 0)

    wVideo = cv.VideoWriter(out_path + "video/" + file_i + "/" + file_i + ".mp4",
                            cv.VideoWriter_fourcc(*'mp4v'), fps, define_size((w, h), size))

    for j in range(total_frames):
        ret, frame = videos['video_' + str(i + 1)].read()
        datum.cvInputData = resize_frame(frame, (w, h), size)

        datum.id = j
        datum.name = f'openpose_{file_i}_{datum.id:09}'
        handler.emplaceAndPop([datum])

        wVideo.write(datum.cvOutputData)
        wVideo.release()

    if predictor == 'openpose':
        handler.stop()


def process_video(predictor=None, in_path='videos', out_path='out'):
    if predictor is None:
        return

    today = date.today()
    d1 = today.strftime("%d.%m.%Y")

    video_files = read_video_files(in_path)
    videos = generate_video_capture(video_files)

    root_path = out_path + '/' + predictor + '_' + d1 + '/'

    if not os.path.exists(root_path):
        os.mkdir(root_path)

    for i, file_i in enumerate(video_files):
        file_i = os.path.splitext(os.path.split(file_i)[1])[0]

        if not os.path.exists(root_path + "video/"):
            os.mkdir(root_path + "video/")

        if not os.path.exists(root_path + "json/"):
            os.mkdir(root_path + "json/")

        if not os.path.exists(root_path + "video/" + file_i):
            os.mkdir(root_path + "video/" + file_i)

        if not os.path.exists(root_path + "json/" + file_i):
            os.mkdir(root_path + "json/" + file_i)

        infere(predictor, i, file_i, root_path, videos, (800, 600))


process_video('openpose', '/media/solid/Ptasek twardy dysk/TFG/videos_test')
