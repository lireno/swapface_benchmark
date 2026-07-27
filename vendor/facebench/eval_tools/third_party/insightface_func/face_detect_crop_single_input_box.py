'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 16:46:04
Description: 
'''
from __future__ import division
import collections
import numpy as np
import glob
import os
import os.path as osp
import cv2
from insightface.model_zoo import model_zoo
from insightface_func.utils import face_align_ffhqandnewarc as face_align

__all__ = ['Face_detect_crop', 'Face']

Face = collections.namedtuple('Face', [
    'bbox', 'kps', 'det_score', 'embedding', 'gender', 'age',
    'embedding_norm', 'normed_embedding',
    'landmark'
])

Face.__new__.__defaults__ = (None, ) * len(Face._fields)


def _prepare_detection_model(model, ctx_id, det_size, det_thresh):
    try:
        model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
    except TypeError:
        model.prepare(ctx_id, input_size=det_size)


def _detect_faces(det_model, img, det_thresh, max_num):
    try:
        return det_model.detect(
            img,
            threshold=det_thresh,
            max_num=max_num,
            metric='default',
        )
    except TypeError as error:
        if 'threshold' not in str(error):
            raise
    try:
        return det_model.detect(
            img,
            thresh=det_thresh,
            max_num=max_num,
            metric='default',
        )
    except TypeError as error:
        if 'thresh' not in str(error):
            raise
    return det_model.detect(img, max_num=max_num, metric='default')


class Face_detect_crop:
    def __init__(self, name, root='~/.insightface_func/models'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                #print('ignore:', onnx_file)
                continue
            model = model_zoo.get_model(onnx_file)
            if model.taskname not in self.models:
                print('find model:', onnx_file, model.taskname)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode ='None'):
        self.det_thresh = det_thresh
        self.mode = mode
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                _prepare_detection_model(model, ctx_id, det_size, det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img, crop_size, max_num=0, input_bbox=None):
        # 如果给了input_bbox，就将其他区域置为0，再送去检测，避免检测到其他人
        if input_bbox is not None:
            x1, y1, x2, y2 = map(int, input_bbox[:4])
            # 创建一个与原图大小相同的黑色背景
            masked_img = np.zeros_like(img)
            # 将 bbox 区域从原图复制到黑色背景上
            masked_img[y1:y2, x1:x2] = img[y1:y2, x1:x2]

            img_for_detection = masked_img # 使用处理后的图像进行检测
        else: 
            img_for_detection = img

        # self.det_thresh = 0.0
        bboxes, kpss = _detect_faces(self.det_model, img_for_detection, self.det_thresh, max_num)

        if bboxes.shape[0] == 0:
            return None
        # ret = []
        # for i in range(bboxes.shape[0]):
        #     bbox = bboxes[i, 0:4]
        #     det_score = bboxes[i, 4]
        #     kps = None
        #     if kpss is not None:
        #         kps = kpss[i]
        #     M, _ = face_align.estimate_norm(kps, crop_size, mode ='None') 
        #     align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
        # for i in range(bboxes.shape[0]):
        #     kps = None
        #     if kpss is not None:
        #         kps = kpss[i]
        #     M, _ = face_align.estimate_norm(kps, crop_size, mode ='None') 
        #     align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)

        det_score = bboxes[..., 4]

        # select the face with the hightest detection score
        best_index = np.argmax(det_score)

        kps = None
        if kpss is not None:
            kps = kpss[best_index]
        M, _ = face_align.estimate_norm(kps, crop_size, mode = self.mode) 
        align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)

        # import pdb; pdb.set_trace()
        
        return [align_img], [M]
