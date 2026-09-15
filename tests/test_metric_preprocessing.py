import ast,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from types import SimpleNamespace
from contextlib import nullcontext
source=Path(__file__).resolve().parents[1]/'vendor/facebench/eval_tools/metrics_calculator_facebench.py'
tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MetricsCalculator')
selected={'extract_gaze','calculate_gaze_cosine_similarity','detect_landmarks_dlib'}
ns={'np':np,'Image':Image,'torch':SimpleNamespace(no_grad=nullcontext)}
for n in cls.body:
 if isinstance(n,ast.FunctionDef) and n.name in selected:exec(compile(ast.Module(body=[n],type_ignores=[]),str(source),'exec'),ns)
class Gaze:
 def __init__(self,detector):self.include_detector=detector;self.seen=None
 def predict_gaze(self,img):self.seen=img.copy();return np.array([0.1]),np.array([0.2])
 def step(self,img):p,y=self.predict_gaze(img);return SimpleNamespace(pitch=p,yaw=y)
class Tests(unittest.TestCase):
 def test_rgb_direct(self):
  model=Gaze(False);obj=SimpleNamespace(gaze_model=model);img=np.array([[[250,20,10]]],dtype=np.uint8)
  ns['extract_gaze'](obj,img);np.testing.assert_array_equal(model.seen,img)
  ns['extract_gaze'](obj,Image.fromarray(img));np.testing.assert_array_equal(model.seen,img)
 def test_bgr_detector(self):
  model=Gaze(True);img=np.array([[[250,20,10]]],dtype=np.uint8)
  ns['extract_gaze'](SimpleNamespace(gaze_model=model),img);np.testing.assert_array_equal(model.seen,img[...,::-1])
 def test_direction_cosine(self):
  f=ns['calculate_gaze_cosine_similarity'];self.assertAlmostEqual(f(None,[1,0],[-1,0]),np.cos(np.deg2rad(2)))
  self.assertAlmostEqual(f(None,[0,0],[0,0]),1)
  self.assertAlmostEqual(f(None,[0,0],[0,180]),-1)
  self.assertIsNone(f(None,None,[0,0]))
 def test_standard_points_and_failure(self):
  pts=np.array([[[1,2],[3,2],[2,3],[1,4],[3,4]]],dtype=np.float32)
  detector=SimpleNamespace(detect=lambda *a,**kw:(np.zeros((1,5)),pts))
  obj=SimpleNamespace(deep3d_landmark_detector=detector)
  np.testing.assert_array_equal(ns['detect_landmarks_dlib'](obj,None),pts[0])
  detector.detect=lambda *a,**kw:(np.zeros((0,5)),None)
  self.assertIsNone(ns['detect_landmarks_dlib'](obj,None))
if __name__=='__main__':unittest.main()


def test_pose_rgb_array_is_not_channel_swapped():
 import torch
 pose_node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='extract_pose')
 scope={'np':np,'Image':Image,'torch':torch,'F':torch.nn.functional}
 exec(compile(ast.Module(body=[pose_node],type_ignores=[]),str(source),'exec'),scope)
 seen=[]
 def transform(img):
  seen.append(np.asarray(img));return torch.zeros(3,2,2)
 obj=SimpleNamespace(pose_model=lambda t:(torch.zeros(1,66),)*3,pose_transform=transform,
  device='cpu',idx_tensor=torch.arange(66))
 rgb=np.array([[[250,20,10]]],dtype=np.uint8)
 scope['extract_pose'](obj,rgb)
 np.testing.assert_array_equal(seen[0],rgb)


def test_cosface_reuses_generated_features_and_preserves_missing_faces():
 node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='calculate_video_face_similarities')
 scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
 calls=[]
 def encode(img,flip):calls.append(img);return img
 obj=SimpleNamespace(detect_and_align_face=lambda img:img,extract_face_feature=encode,
                     calculate_cosine_similarity=lambda a,b:float(a*b))
 outputs=scope['calculate_video_face_similarities'](obj,[1.,None,2.],[3.,4.])
 assert len(calls)==4  # two references plus two valid generated frames, not six
 assert outputs==[(4.5,[3.,None,6.]),(6.,[4.,None,8.])]
