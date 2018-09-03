import argparse
import cv2
import json
import numpy as np
import os

from mean_average_precision.detection_map import DetectionMAP
from mean_average_precision.utils.show_frame import show_frame
import numpy as np
import matplotlib.pyplot as plt


CLASS_ID = {'BG': 0, 'pedestrian': 1}

def bb_normalize(bb_list, file_id):
    if file_id.startswith('c6'):
        bb_list[:,0] /= 720; bb_list[:,2] /= 720
        bb_list[:,1] /= 576; bb_list[:,3] /= 576
    else:
        bb_list[:,0] /= 1920; bb_list[:,2] /= 1920
        bb_list[:,1] /= 1080; bb_list[:,3] /= 1080

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', help='The output file of model evaluation.')
    parser.add_argument('--gt_dir', help='The directory holding the gt files.')
    args = parser.parse_args()

    with open(args.output_dir, 'r') as f:
        output = json.load(f)
    fname_list = []
    for item in output:
        fname_list.append(os.path.basename(item[0]).split('.')[0])
    fname_list_sorted = sorted(fname_list)
    fname_list_sorted_index = sorted(range(len(fname_list)), key=lambda k: fname_list[k])

    gtfname_list = os.listdir(args.gt_dir)
    for idx, item in enumerate(gtfname_list):
        gtfname_list[idx] = item.split('.')[0]
    gtfname_list_sorted = sorted(gtfname_list)

    frames = []
    for idx, gtfname in enumerate(gtfname_list_sorted):
        """
            pred[0]: abs fname
            pred[1]: bb list
            pred[2]: label list
            pred[3]: score list
        """
        pred = output[fname_list_sorted_index[idx]]
        file_id = os.path.basename(pred[0]).split('.')[0]
        # assert file_id == gtfname

        pred_bb = np.array(pred[1], dtype=np.float32)
        if pred_bb.any():
            bb_normalize(pred_bb, file_id)

        pred_cls = np.array(pred[2], dtype=np.int16)
        pred_conf = np.array(pred[3], dtype=np.float32)

        gt_bb = []
        gt_cls = []
        with open(os.path.join(args.gt_dir, gtfname) + '.txt', 'r') as f:
            for line in f:
                line = line.split()
                gt_bb.append(list(map(float,line[1:])))
                gt_cls.append(CLASS_ID[line[0]])
        gt_bb = np.array(gt_bb, dtype=np.float32)
        bb_normalize(gt_bb, file_id)
        gt_cls = np.array(gt_cls, dtype=np.int16)

        frames.append([file_id, (pred_bb, pred_cls, pred_conf, gt_bb, gt_cls)])

    n_class = 2

    mAP = DetectionMAP(n_class, overlap_threshold=0.7)
    for i, frame in enumerate(frames):
        print("Evaluate frame {}".format(i))
        img = cv2.imread(os.path.join(args.gt_dir, '..', 'frames', frame[0]) + '.jpg', cv2.IMREAD_COLOR)
        img = img[:, :, [2, 1, 0]]
        # show_frame(*frame[1], background=img)
        mAP.evaluate(*frame[1])

    mAP.plot()
    plt.show()