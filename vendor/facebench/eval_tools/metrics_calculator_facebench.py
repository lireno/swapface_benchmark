# modified from CosFace_pytorch/lfw_eval.py and deep-head-pose/code/test_on_video_dlib.py

import torch
import numpy as np
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
from tqdm import tqdm
import sys
import os
import glob
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter
from typing import List, Optional, Tuple
import cv2

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/root/humanvid/cache/torch_extensions")
for _plugin_name in ("nvdiffrast_plugin", "nvdiffrast_plugin_gl"):
    _plugin_dir = os.path.join(os.environ["TORCH_EXTENSIONS_DIR"], _plugin_name)
    if _plugin_dir not in sys.path:
        sys.path.insert(0, _plugin_dir)

# 获取当前文件的目录，用于构建绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 添加warping error模块路径
sys.path.append(project_root)
from warping_error import compute_ref_warping_error

# 添加face模型路径
cosface_path = os.path.join(project_root, "eval_tools/third_party/CosFace_pytorch")
sys.path.append(cosface_path)

# 添加pose模型路径
pose_path = os.path.join(project_root, "eval_tools/third_party/deep_head_pose/code")
sys.path.append(pose_path)

# 添加insightface_func路径
insightface_path = os.path.join(project_root, "eval_tools/third_party")
sys.path.append(insightface_path)

from CosFace_pytorch import net
import hopenet
from insightface_func.face_detect_crop_single import Face_detect_crop  # TODO: 支持最新版的insightface

# 添加gaze相关导入
try:
    from l2cs import Pipeline, render
except ImportError:
    print("Warning: l2cs not found. Gaze metric will not be available.")
    Pipeline = None
    render = None

# 添加Deep3DFaceRecon_pytorch相关导入
# Deep3DFaceRecon_pytorch 路径
deep3d_path = os.path.join(project_root, "eval_tools", "third_party", "Deep3DFaceRecon_pytorch")
sys.path.append(deep3d_path)

DEEP3D_AVAILABLE = True
SHOW_INNER_PROGRESS = False

# ----------- 模块检查 -------------
try:
    from models import create_model
except ImportError as e:
    print("[ImportError] models/create_model 导入失败:", e)
    DEEP3D_AVAILABLE = False

try:
    from util.load_mats import load_lm3d
except ImportError as e:
    print("[ImportError] util/load_mats 导入失败:", e)
    DEEP3D_AVAILABLE = False

try:
    from util.preprocess import align_img
except ImportError as e:
    print("[ImportError] util/preprocess 导入失败:", e)
    DEEP3D_AVAILABLE = False

try:
    import dlib
except ImportError as e:
    print("[ImportError] dlib 导入失败:", e)
    DEEP3D_AVAILABLE = False

# ----------- 总结 -------------
if DEEP3D_AVAILABLE:
    print("✅ Deep3DFaceRecon_pytorch 及依赖全部可用")
else:
    print("⚠️ Deep3DFaceRecon_pytorch 或依赖缺失，请根据上面报错检查路径和安装情况")

import cv2

def get_model_device(model):
    """显示模型所在的设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None

def get_current_process_gpus():
    """
    获取当前进程可见的 GPU ID 列表（整数形式）。
    依赖 CUDA_VISIBLE_DEVICES 环境变量。
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        return []  # 没有限制，说明可能能看见所有 GPU
    try:
        return [int(x) for x in cuda_visible.split(",") if x.strip() != ""]
    except ValueError:
        return []

def get_torch_current_device():
    if torch.cuda.is_available():
        return torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        return None, "CPU only"

class MetricsCalculator:
    def __init__(self,device) -> None:
        # Ray会自动分配CUDA设备，无需手动指定device
        # breakpoint()
        self.device = device
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(self.device)
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        # print(f"LPIPS model load on {get_model_device(self.lpips_metric_calculator)}")
        # print(f"SSIM model load on {get_model_device(self.ssim_metric_calculator)}")
        # print(f"Current process visible GPUs: {get_current_process_gpus()}")
        # print(f"Torch current device: {get_torch_current_device()}")
        
        # 初始化人脸特征提取模型
        self.face_model = None
        self.face_model_path = None
        self.face_transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
        
        # 初始化姿态估计模型
        self.pose_model = None
        self.pose_model_path = None
        self.pose_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.video_pose_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 初始化姿态预测的索引张量
        self.idx_tensor = torch.FloatTensor([idx for idx in range(66)]).to(self.device)

        # 初始化人脸检测对齐模型
        self.face_detect_align_model = None
        self.face_detect_align_model_path = None
        
        # 初始化gaze估计模型
        self.gaze_model = None
        self.gaze_model_path = None
        
        # 初始化Deep3DFaceRecon模型
        self.deep3d_model = None
        self.deep3d_model_path = None
        self.dlib_detector = None
        self.dlib_predictor = None
        self.lm3d_std = None
    
    def load_face_model(self, model_path):
        """加载人脸特征提取模型
        
        Args:
            model_path: 模型权重路径
        """
        if self.face_model is None or model_path != self.face_model_path:
            self.face_model = net.sphere()
            self.face_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.face_model.to(self.device)
            self.face_model.eval()
            self.face_model_path = model_path
        # print(f"face model load on {get_model_device(self.face_model)}")
            
    def load_pose_model(self, model_path):
        """加载姿态估计模型
        
        Args:
            model_path: 模型权重路径
        """
        if self.pose_model is None or model_path != self.pose_model_path:
            self.pose_model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
            saved_state_dict = torch.load(model_path, map_location='cpu')
            self.pose_model.load_state_dict(saved_state_dict)
            self.pose_model.to(self.device)
            self.pose_model.eval()
            self.pose_model_path = model_path
        # print(f"pose model load on {get_model_device(self.pose_model)}")
    
    def load_face_detect_align_model(self, model_path='./insightface_func/models'):
        """加载人脸检测对齐模型
        
        Args:
            model_path: 模型权重路径
        """
        if self.face_detect_align_model is None or model_path != self.face_detect_align_model_path:
            self.face_detect_align_model = Face_detect_crop(name='antelope', root=model_path)
            # NOTE 这里的阈值设置得低一些以尽可能地能检测到人脸，但注意不能设成0，要不然box太多numpy后处理会很卡
            self.face_detect_align_model.prepare(ctx_id=0, det_thresh=0.1, det_size=(640,640), mode='None')
            # self.face_detect_align_model.prepare(ctx_id=0, det_thresh=0.01, det_size=(640,640), mode='None')
            self.face_detect_align_model_path = model_path
    
    def load_gaze_model(self, model_path):
        """加载gaze估计模型
        
        Args:
            model_path: gaze模型权重路径
        """
        if Pipeline is None:
            raise ImportError("l2cs not available. Please install l2cs-net.")
        
        if self.gaze_model is None or model_path != self.gaze_model_path:
            # FaceBench already crops faces from masks before gaze evaluation.
            # Some l2cs releases construct an internal RetinaFace with a gpu_id
            # argument that is incompatible with the face_detection package in
            # this environment, so avoid the redundant detector here.
            self.gaze_model = Pipeline(
                weights=model_path,
                arch='ResNet50',
                device=self.device,
                include_detector=False,
            )
            if not hasattr(self.gaze_model, 'softmax'):
                self.gaze_model.softmax = torch.nn.Softmax(dim=1)
            if not hasattr(self.gaze_model, 'idx_tensor'):
                self.gaze_model.idx_tensor = torch.arange(
                    90, dtype=torch.float32, device=self.device
                )
            self.gaze_model_path = model_path
            
            print(f"Loading gaze model to device: {self.device}")
            self._ensure_gaze_model_device()
            print(f"Gaze model loaded and all components moved to device: {self.device}")
    
    def _ensure_gaze_model_device(self):
        """确保gaze模型的所有组件都在正确的device上"""
        if self.gaze_model is None:
            return
            
        # 设置Pipeline的device属性
        if hasattr(self.gaze_model, 'device'):
            self.gaze_model.device = self.device
        
        # 检查并设置gaze模型内部组件的device
        if hasattr(self.gaze_model, 'model'):
            if hasattr(self.gaze_model.model, 'to'):
                self.gaze_model.model.to(self.device)
            if hasattr(self.gaze_model.model, 'device'):
                self.gaze_model.model.device = self.device
        
        # 确保所有相关的tensor组件在正确的device上
        for attr_name in ['model', 'net', 'detector', 'face_detector']:
            if hasattr(self.gaze_model, attr_name):
                attr_obj = getattr(self.gaze_model, attr_name)
                if hasattr(attr_obj, 'to'):
                    attr_obj.to(self.device)
    
    def load_deep3d_model(self, checkpoints_dir=None, bfm_folder=None):
        """加载Deep3DFaceRecon模型
        
        Args:
            checkpoints_dir: 模型checkpoint目录路径
            bfm_folder: BFM模型文件夹路径
        """
        if not DEEP3D_AVAILABLE:
            raise ImportError("Deep3DFaceRecon_pytorch not available.")
        
        if checkpoints_dir is None:
            checkpoints_dir = os.path.join(project_root, "eval_tools/third_party/Deep3DFaceRecon_pytorch/checkpoints")
        if bfm_folder is None:
            bfm_folder = os.path.join(project_root, "eval_tools/third_party/Deep3DFaceRecon_pytorch/BFM")
        
        # 创建选项类
        class InferenceOptions:
            def __init__(self):
                # Basic parameters
                self.name = 'pretrained'
                self.model = 'facerecon'
                self.epoch = '20'
                self.gpu_ids = '0'
                self.checkpoints_dir = checkpoints_dir
                self.bfm_folder = bfm_folder
                
                # Model parameters
                self.net_recon = 'resnet50'
                self.use_last_fc = False
                self.bfm_model = 'BFM_model_front.mat'
                self.phase = 'test'
                self.dataset_mode = None
                self.serial_batches = True
                self.no_flip = True
                self.init_path = os.path.join(checkpoints_dir, 'init_model/resnet50-0676ba61.pth')
                
                # Additional required parameters
                self.isTrain = False
                self.use_ddp = False
                self.verbose = False
                self.suffix = ''
                self.vis_batch_nums = 1
                self.eval_batch_nums = float('inf')
                self.ddp_port = '12355'
                self.display_per_batch = True
                self.add_image = True
                self.world_size = 1
                
                # Face recognition network
                self.net_recog = 'r50'
                self.use_crop_face = True
                self.use_predef_M = False
                
                # Renderer parameters
                self.focal = 1015.
                self.center = 112.
                self.camera_d = 10.
                self.z_near = 5.
                self.z_far = 15.
                self.use_opengl = False
        
        if self.deep3d_model is None or checkpoints_dir != self.deep3d_model_path:
            self.opt = InferenceOptions()
            
            # 创建和设置模型
            self.deep3d_model = create_model(self.opt)
            self.deep3d_model.setup(self.opt)
            self.deep3d_model.device = self.device
            self.deep3d_model.parallelize()
            self.deep3d_model.eval()
            
            # 初始化dlib检测器
            self.dlib_detector = dlib.get_frontal_face_detector()
            predictor_path = os.path.join(checkpoints_dir, 'dlib_predictor_recognition/shape_predictor_5_face_landmarks.dat')
            if os.path.exists(predictor_path):
                self.dlib_predictor = dlib.shape_predictor(predictor_path)
            else:
                print(f"Warning: Dlib predictor not found at {predictor_path}")
                self.dlib_predictor = None
            
            # 加载3D landmarks标准
            self.lm3d_std = load_lm3d(self.opt.bfm_folder)
            
            self.deep3d_model_path = checkpoints_dir
        
        # print(f"Deep3D face reconstruction model loaded on {get_model_device(self.deep3d_model.net_recon)}")
    
    def detect_landmarks_dlib(self, image):
        """使用dlib检测5个关键点
        
        Args:
            image: 输入图像，numpy数组 (BGR格式)
            
        Returns:
            landmarks: numpy数组 (5, 2) 或 None
        """
        if self.dlib_detector is None or self.dlib_predictor is None:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.dlib_detector(gray)
        
        if len(faces) == 0:
            return None
        
        # 使用最大的人脸
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        landmarks = self.dlib_predictor(gray, face)
        
        # 转换为numpy数组 (5个点: 左眼, 右眼, 鼻子, 左嘴角, 右嘴角)
        points = np.zeros((5, 2), dtype=np.float32)
        for i in range(5):
            points[i] = (landmarks.part(i).x, landmarks.part(i).y)
        
        return points
    
    def preprocess_for_deep3d(self, frame, landmarks):
        """为Deep3D模型预处理帧和关键点
        
        Args:
            frame: 输入帧，numpy数组 (BGR格式)
            landmarks: 关键点，numpy数组 (5, 2)
            
        Returns:
            im_tensor: 图像张量
            lm_tensor: 关键点张量
        """
        if self.lm3d_std is None:
            print(f"lm3d_std is None!!!")
            return None, None
        
        # 转换BGR到RGB
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        H, W = frame.shape[:2]
        
        # 准备关键点
        lm = landmarks.copy()
        lm[:, 1] = H - 1 - lm[:, 1]  # 翻转y坐标
        
        # 对齐图像
        
        _, im_aligned, lm_aligned, _ = align_img(im, lm, self.lm3d_std)
        
        
        # 转换为张量
        im_tensor = torch.tensor(np.array(im_aligned)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm_tensor = torch.tensor(lm_aligned).unsqueeze(0)
        
        return im_tensor, lm_tensor
    
    def extract_face_coefficients(self, im_tensor, lm_tensor):
        """提取人脸重建系数
        
        Args:
            im_tensor: 图像张量
            lm_tensor: 关键点张量
            
        Returns:
            coeffs_dict: 包含各种系数的字典
        """
        if self.deep3d_model is None:
            raise ValueError("Deep3D model not loaded. Call load_deep3d_model() first.")
        
        # 准备数据
        data = {
            'imgs': im_tensor.to(self.device),
            'lms': lm_tensor.to(self.device)
        }
        
        # 推理
        self.deep3d_model.set_input(data)
        self.deep3d_model.test()
        
        # 获取系数
        coeffs_dict = self.deep3d_model.pred_coeffs_dict
        
        # 转换为numpy
        coeffs_numpy = {}
        for key, value in coeffs_dict.items():
            coeffs_numpy[key] = value.cpu().numpy().flatten()
        
        return coeffs_numpy
    
    def extract_exp_gamma_from_image(self, img, frame_idx=None):
        """从图像中提取expression和gamma系数
        
        Args:
            img: 输入图像，PIL Image或numpy数组
            frame_idx: 可选，帧序号，用于调试输出
            
        Returns:
            exp_coeff: expression系数，numpy数组
            gamma_coeff: gamma系数，numpy数组
        """
        prefix = f"[Frame {frame_idx}] " if frame_idx is not None else ""

        if self.deep3d_model is None:
            raise ValueError("Deep3D model not loaded. Call load_deep3d_model() first.")
        
        # 转换图像为numpy数组（BGR格式）
        try:
            if isinstance(img, Image.Image):
                img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # print(prefix + f"输入为 PIL.Image，转换为 numpy，shape={img_np.shape}")
            elif isinstance(img, np.ndarray):
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img_np = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    # print(prefix + f"输入为 numpy RGB，转换为 BGR，shape={img_np.shape}")
                else:
                    img_np = img
                    # print(prefix + f"输入为 numpy 非3通道，直接使用，shape={img_np.shape}")
            else:
                raise ValueError("Unsupported image format")
        except Exception as e:
            print(prefix + f"图像转换失败: {e}")
            return None, None
        
        # 检测关键点
        landmarks = self.detect_landmarks_dlib(img_np)
        if landmarks is None:
            return None, None

        # 预处理
        im_tensor, lm_tensor = self.preprocess_for_deep3d(img_np, landmarks)

        # 提取系数
        try:
            coeffs = self.extract_face_coefficients(im_tensor, lm_tensor)
            if coeffs is None:
                print(prefix + "extract_face_coefficients 返回 None")
                return None, None

            exp_coeff = coeffs.get('exp', None)
            gamma_coeff = coeffs.get('gamma', None)

            # print(prefix + f"提取系数: exp={None if exp_coeff is None else exp_coeff.shape}, "
            #     f"gamma={None if gamma_coeff is None else gamma_coeff.shape}")

            return exp_coeff, gamma_coeff
        except Exception as e:
            print(prefix + f"Error extracting coefficients: {e}")
            return None, None
        
    def calculate_exp_gamma_l2_distance(self, exp1, gamma1, exp2, gamma2, frame_idx=None):
        """计算exp和gamma系数的L2距离"""
        exp_l2_dist = None
        gamma_l2_dist = None

        if exp1 is None or exp2 is None:
            print(f"[Frame {frame_idx}] exp1 或 exp2 为 None, 跳过计算")
        else:
            try:
                exp_l2_dist = np.linalg.norm(exp1 - exp2)
                # print(f"[Frame {frame_idx}] exp_l2_dist = {exp_l2_dist}")
            except Exception as e:
                print(f"[Frame {frame_idx}] 计算 exp_l2_dist 出错: {e}, exp1={type(exp1)}, exp2={type(exp2)}")

        if gamma1 is None or gamma2 is None:
            print(f"[Frame {frame_idx}] gamma1 或 gamma2 为 None, 跳过计算")
        else:
            try:
                gamma_l2_dist = np.linalg.norm(gamma1 - gamma2)
                # print(f"[Frame {frame_idx}] gamma_l2_dist = {gamma_l2_dist}")
            except Exception as e:
                print(f"[Frame {frame_idx}] 计算 gamma_l2_dist 出错: {e}, gamma1={type(gamma1)}, gamma2={type(gamma2)}")

        return exp_l2_dist, gamma_l2_dist


    def calculate_video_exp_gamma_distance(self, video_frames1, video_frames2):
        """计算两个视频序列的exp和gamma距离"""
        assert len(video_frames1) == len(video_frames2), "Video frame lists should have the same length."

        exp_l2_distances = []
        gamma_l2_distances = []

        for i in tqdm(range(len(video_frames1)), desc="Calculating exp and gamma L2 distance", disable=not SHOW_INNER_PROGRESS):
            frame1 = video_frames1[i]
            frame2 = video_frames2[i]

            # 提取exp和gamma系数
            exp1, gamma1 = self.extract_exp_gamma_from_image(frame1)
            exp2, gamma2 = self.extract_exp_gamma_from_image(frame2)
            # print(f"[Frame {i}] 提取到 exp1={None if exp1 is None else exp1.shape}, "
            #     f"gamma1={None if gamma1 is None else gamma1.shape}, "
            #     f"exp2={None if exp2 is None else exp2.shape}, "
            #     f"gamma2={None if gamma2 is None else gamma2.shape}")

            # 计算L2距离
            exp_l2_dist, gamma_l2_dist = self.calculate_exp_gamma_l2_distance(exp1, gamma1, exp2, gamma2, frame_idx=i)

            if exp_l2_dist is not None:
                exp_l2_distances.append(exp_l2_dist)
            if gamma_l2_dist is not None:
                gamma_l2_distances.append(gamma_l2_dist)

        # 计算平均距离
        avg_exp_l2_dist = np.mean(exp_l2_distances) if exp_l2_distances else None
        avg_gamma_l2_dist = np.mean(gamma_l2_distances) if gamma_l2_distances else None

        # print("=== 计算结果汇总 ===")
        # print(f"exp 距离数量: {len(exp_l2_distances)}, 平均值: {avg_exp_l2_dist}")
        # print(f"gamma 距离数量: {len(gamma_l2_distances)}, 平均值: {avg_gamma_l2_dist}")

        return avg_exp_l2_dist, avg_gamma_l2_dist, exp_l2_distances, gamma_l2_distances
    
    def detect_and_align_face(self, img, crop_size=(112, 112), output_size=(112, 96)): 
        """检测并对齐人脸
        
        Args:
            img: 输入图像，PIL Image或numpy数组
            crop_size: 对齐后的人脸图像大小
            
        Returns:
            对齐后的人脸图像
        
        注意这里crop_size为(112,112)，而cosface需要(112,96)，因此使用crop的方法调整

        """
        if self.face_detect_align_model is None:
            raise ValueError("Face detection model not loaded. Call load_face_detect_align_model first.")
        
        # 转换图像为PIL Image类型
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        
        # 转换为模型输入格式
        img = np.array(img)[..., ::-1]  # 对齐cv2格式
        
        # 检测人脸并对齐
        res = self.face_detect_align_model.get(img, crop_size=crop_size)

        if res is None:
            # print("No face detected.")
            # return None
            # TODO 优化这种检测不到的情况
            print("No face detected. use input image instead.")
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = img.resize((output_size[1], output_size[0]), Image.BILINEAR)  # PIL的size是(w, h)
            return img

        align_img = res[0]

        # crop align image to output size
        crop_x = (crop_size[0] - output_size[0]) // 2
        crop_y = (crop_size[1] - output_size[1]) // 2
        align_img = align_img[0][crop_x:crop_x + output_size[0], crop_y:crop_y + output_size[1], :]
        
        align_img_pil = Image.fromarray(cv2.cvtColor(align_img,cv2.COLOR_BGR2RGB))
        
        # 返回对齐后的人脸图像
        return align_img_pil
    
    def extract_face_feature(self, img, use_flip=False):
        """提取人脸特征
        
        Args:
            img: 输入图像，PIL Image或numpy数组
            
        Returns:
            人脸特征向量
        """
        if self.face_model is None:
            raise ValueError("Face model not loaded. Call load_face_model first.")
        
        # 转换图像为PIL Image类型
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))

        # import pdb; pdb.set_trace()
        
        # 检查尺寸是否为112x96，否则resize
        if img.size != (96, 112):  # PIL的size是(w, h)
            # img = img.resize((96, 112), Image.BILINEAR)
            raise ValueError("Image size should be (96, 112). Current size: {}".format(img.size))

        
        # 转换为模型输入格式并提取特征
        with torch.no_grad():
            img_tensor = self.face_transform(img).unsqueeze(0).to(self.device)
            if use_flip:
                img_flipped_tensor = self.face_transform(img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(self.device)
                feat = torch.cat((self.face_model(img_tensor), self.face_model(img_flipped_tensor)), 1)[0].cpu()
            else:
                feat = self.face_model(img_tensor)[0].cpu()
        
        return feat
    
    def extract_pose(self, img):
        """提取人脸姿态 (yaw, pitch, roll)
        
        Args:
            img: 输入图像，PIL Image或numpy数组
            
        Returns:
            姿态元组 (yaw, pitch, roll)，单位为度
        """
        if self.pose_model is None:
            raise ValueError("Pose model not loaded. Call load_pose_model first.")
            
        # 转换图像为PIL Image类型
        if isinstance(img, np.ndarray):
            if img.shape[2] == 3: # 如果是BGR格式（OpenCV格式）
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img.astype(np.uint8))
        
        # 转换为模型输入格式并提取姿态
        with torch.no_grad():
            img_tensor = self.pose_transform(img).unsqueeze(0).to(self.device)
            yaw, pitch, roll = self.pose_model(img_tensor)
            
            # 将softmax应用于输出以获得概率分布
            yaw_predicted = F.softmax(yaw, dim=1)
            pitch_predicted = F.softmax(pitch, dim=1)
            roll_predicted = F.softmax(roll, dim=1)
            
            # 计算连续的角度值（单位：度）
            yaw_value = torch.sum(yaw_predicted[0] * self.idx_tensor) * 3 - 99
            pitch_value = torch.sum(pitch_predicted[0] * self.idx_tensor) * 3 - 99
            roll_value = torch.sum(roll_predicted[0] * self.idx_tensor) * 3 - 99
            
        return yaw_value.cpu().item(), pitch_value.cpu().item(), roll_value.cpu().item()
    
    def extract_pose_video(self, video):
        """提取人脸姿态 (yaw, pitch, roll)
        
        Args:
            video: numpy 数组 （N, H, W, C）
            
        Returns:
            姿态元组 (yaw, pitch, roll)，单位为度
        """
        if self.pose_model is None:
            raise ValueError("Pose model not loaded. Call load_pose_model first.")
            
        # 转换图像为PIL Image类型
        if not isinstance(video, np.ndarray):
            return None
        
        # 转换为模型输入格式并提取姿态
        with torch.no_grad():
            video_tensor = self.video_pose_transform((torch.from_numpy(video).permute(0,3,1,2).float() / 255.0)).to(self.device)
            yaw, pitch, roll = self.pose_model(video_tensor)
            
            # 将softmax应用于输出以获得概率分布
            yaw_predicted = F.softmax(yaw, dim=1)
            pitch_predicted = F.softmax(pitch, dim=1)
            roll_predicted = F.softmax(roll, dim=1)
            
            # 计算连续的角度值（单位：度）
            yaw_value = torch.sum(yaw_predicted * self.idx_tensor, dim=1) * 3 - 99
            pitch_value = torch.sum(pitch_predicted * self.idx_tensor, dim=1) * 3 - 99
            roll_value = torch.sum(roll_predicted * self.idx_tensor, dim=1) * 3 - 99
        
        # 5. 将结果传回 CPU
        results = []
        for y, p, r in zip(yaw_value.cpu(), pitch_value.cpu(), roll_value.cpu()):
            results.append((y.item(), p.item(), r.item()))
            
        # return yaw_value.item(), pitch_value.item(), roll_value.item()
        return results
    
    def calculate_cosine_similarity(self, feat1, feat2):
        """计算两个特征向量的余弦相似度
        
        Args:
            feat1: 特征向量1
            feat2: 特征向量2
            
        Returns:
            余弦相似度，范围[-1, 1]
        """
        sim = feat1.dot(feat2) / (feat1.norm() * feat2.norm() + 1e-5)
        return sim.item()
    
    def calculate_pose_l2_distance(self, pose1, pose2):
        """计算两个姿态之间的L2距离
        
        Args:
            pose1: 第一个姿态 (yaw, pitch, roll)
            pose2: 第二个姿态 (yaw, pitch, roll)
            
        Returns:
            L2距离
        """
        yaw1, pitch1, roll1 = pose1
        yaw2, pitch2, roll2 = pose2
        
        # 计算各角度差的平方和的平方根
        l2_dist = np.sqrt((yaw1 - yaw2)**2 + (pitch1 - pitch2)**2 + (roll1 - roll2)**2)
        return l2_dist
    
    def extract_gaze(self, img):
        """提取视线方向 (pitch, yaw)
        
        Args:
            img: 输入图像，PIL Image或numpy数组
            
        Returns:
            视线方向numpy数组 [pitch, yaw]，单位为度，如果检测失败返回None
        """
        if self.gaze_model is None:
            raise ValueError("Gaze model not loaded. Call load_gaze_model first.")
        
        # 转换图像为numpy数组（BGR格式，因为l2cs期望BGR格式）
        if isinstance(img, Image.Image):
            img = np.array(img)[..., ::-1]  # RGB -> BGR
        elif isinstance(img, np.ndarray):
            if img.shape[2] == 3:  # 如果是RGB格式
                img = img[..., ::-1]  # RGB -> BGR
        
        # 使用gaze模型进行预测
        try:
            with torch.no_grad():
                # 确保模型在正确的device上（调用辅助函数）
                # self._ensure_gaze_model_device()
                
                if getattr(self.gaze_model, 'include_detector', True):
                    results = self.gaze_model.step(img)
                    pitch = results.pitch
                    yaw = results.yaw
                else:
                    pitch, yaw = self.gaze_model.predict_gaze(img)

                if len(pitch) > 0 and len(yaw) > 0:
                    # l2cs返回的是弧度制，转换为度并返回numpy数组
                    pitch_deg = float(pitch[0]) * 180.0 / np.pi
                    yaw_deg = float(yaw[0]) * 180.0 / np.pi
                    return np.array([pitch_deg, yaw_deg])
                else:
                    print("No face detected for gaze estimation.")
                    return None
        except Exception as e:
            print(f"Error in gaze estimation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_gaze_l2_distance(self, gaze1, gaze2):
        """计算两个视线方向之间的L2距离
        
        Args:
            gaze1: 第一个视线方向 numpy数组 [pitch, yaw] 或 None
            gaze2: 第二个视线方向 numpy数组 [pitch, yaw] 或 None
            
        Returns:
            L2距离
        """
        if gaze1 is None or gaze2 is None:
            return None
        
        # 计算各角度差的平方和的平方根
        l2_dist = np.linalg.norm(gaze1 - gaze2)
        return l2_dist
    
    def calculate_gaze_cosine_similarity(self, gaze1, gaze2):
        """计算两个视线方向向量的余弦相似度
        
        Args:
            gaze1: 第一个视线方向 numpy数组 [pitch, yaw] 或 None
            gaze2: 第二个视线方向 numpy数组 [pitch, yaw] 或 None
            
        Returns:
            余弦相似度，范围[-1, 1]
        """
        if gaze1 is None or gaze2 is None:
            return None
        
        # 计算余弦相似度
        dot_product = np.dot(gaze1, gaze2)
        norm1 = np.linalg.norm(gaze1)
        norm2 = np.linalg.norm(gaze2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return similarity
    
    def calculate_face_similarity(self, img1, img2, use_flip=True):
        """计算两张图像的人脸相似度
        
        Args:
            img1: 图像1
            img2: 图像2
            
        Returns:
            人脸相似度分数
        """

        # 检测并对齐人脸
        aligned_face1 = self.detect_and_align_face(img1)
        if aligned_face1 is None:
            print("No face detected.")
        else:
            img1 = aligned_face1

        aligned_face2 = self.detect_and_align_face(img2)
        if aligned_face2 is None:
            print("No face detected.")
        else:
            img2 = aligned_face2

        feat1 = self.extract_face_feature(img1, use_flip)
        feat2 = self.extract_face_feature(img2, use_flip)
        return self.calculate_cosine_similarity(feat1, feat2)
    
    def calculate_image_pose_similarity(self, img1, img2):
        """计算两张图像的姿态相似度（L2距离）
        
        Args:
            img1: 图像1
            img2: 图像2
            
        Returns:
            姿态L2距离
        """
        pose1 = self.extract_pose(img1)
        pose2 = self.extract_pose(img2)
        return self.calculate_pose_l2_distance(pose1, pose2)
    
    def calculate_video_pose_distance(self, img_list1, img_list2):
        """计算两张图像的姿态距离（与calculate_image_pose_similarity相同）
        
        Args:
            img1: 图像1
            img2: 图像2
            
        Returns:
            姿态L2距离
        """
        assert len(img_list1) == len(img_list2), "Image lists should have the same length."

        pose_l2_distances = []
        for i in range(len(img_list1)):
            pose1 = self.extract_pose(img_list1[i])
            pose2 = self.extract_pose(img_list2[i])
            pose_l2_dist = self.calculate_pose_l2_distance(pose1, pose2)
            pose_l2_distances.append(pose_l2_dist)

        return np.mean(pose_l2_distances), pose_l2_distances

    def calculate_video_gaze_distance(self, img_list1, img_list2):
        """计算两个视频帧序列的视线距离
        
        Args:
            img_list1: 第一个视频帧列表
            img_list2: 第二个视频帧列表
            
        Returns:
            平均视线L2距离和每一帧的视线L2距离列表
        """
        assert len(img_list1) == len(img_list2), "Image lists should have the same length."

        gaze_l2_distances = []
        gaze_cosine_similarities = []
        
        for i in tqdm(range(len(img_list1)), desc="Calculating gaze distance", disable=not SHOW_INNER_PROGRESS):
            gaze1 = self.extract_gaze(img_list1[i])
            gaze2 = self.extract_gaze(img_list2[i])
            # print(f"gaze1: {gaze1}, gaze2: {gaze2}")
            
            l2_dist = self.calculate_gaze_l2_distance(gaze1, gaze2)
            cosine_sim = self.calculate_gaze_cosine_similarity(gaze1, gaze2)
            
            if l2_dist is not None:
                gaze_l2_distances.append(l2_dist)
            if cosine_sim is not None:
                gaze_cosine_similarities.append(cosine_sim)

        avg_l2_dist = np.mean(gaze_l2_distances) if gaze_l2_distances else None
        avg_cosine_sim = np.mean(gaze_cosine_similarities) if gaze_cosine_similarities else None
        
        return avg_l2_dist, gaze_l2_distances, avg_cosine_sim, gaze_cosine_similarities

    def calculate_video_face_similarity(self, video_frames, reference_img, use_flip=True):
        """计算视频中每帧与参考图像的人脸相似度
        
        Args:
            video_frames: 视频帧列表，每个元素为一帧图像
            reference_img: 参考图像
            
        Returns:
            平均人脸相似度和每一帧的相似度列表
        """

        # 检测并对齐人脸
        aligned_face = self.detect_and_align_face(reference_img)
        if aligned_face is None:
            print("No face detected.")
        else:
            reference_img = aligned_face

        # 提取参考图像的特征
        ref_feat = self.extract_face_feature(reference_img, use_flip)
        
        similarity_scores = []
        
        # 计算每一帧的相似度
        for i in tqdm(range(len(video_frames)), desc="Calculating face similarity", disable=not SHOW_INNER_PROGRESS):
            frame = video_frames[i]

            # 检测并对齐人脸
            aligned_face = self.detect_and_align_face(frame)
            if aligned_face is None:
                print("No face detected.")
            else:
                frame = aligned_face

            frame_feat = self.extract_face_feature(frame, use_flip)
            sim_score = self.calculate_cosine_similarity(frame_feat, ref_feat)
            similarity_scores.append(sim_score)
        
        # 计算平均相似度
        avg_similarity = np.mean(similarity_scores)
        
        return avg_similarity, similarity_scores
    
    def calculate_video_pose_similarity(self, video_frames_pred, video_frames_gt):
        """计算视频中每帧的姿态L2距离
        
        Args:
            video_frames_pred: 预测视频帧列表
            video_frames_gt: 真实视频帧列表
            
        Returns:
            平均姿态L2距离和每一帧的姿态L2距离列表
        """
        assert len(video_frames_pred) == len(video_frames_gt), "Videos should have the same number of frames."
        
        pose_l2_distances = []
        
        for i in tqdm(range(len(video_frames_pred)), desc="Calculating pose L2 distance", disable=not SHOW_INNER_PROGRESS):
            frame_pred = video_frames_pred[i]
            frame_gt = video_frames_gt[i]
            
            pose_l2_dist = self.calculate_image_pose_similarity(frame_pred, frame_gt)
            pose_l2_distances.append(pose_l2_dist)
        
        # 计算平均姿态L2距离
        avg_pose_l2_dist = np.mean(pose_l2_distances)
        
        return avg_pose_l2_dist, pose_l2_distances
    
    def calculate_lpips(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        """计算单张图像的LPIPS
        
        Args:
            img_pred: 预测图像
            img_gt: 真实图像
            mask_pred: 预测图像的蒙版
            mask_gt: 真实图像的蒙版
        
        Returns:
            LPIPS分数
        """
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32) / 255
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32) / 255
            img_gt = img_gt * mask_gt
            
        img_pred_tensor = torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
            
        score = self.lpips_metric_calculator(img_pred_tensor*2-1, img_gt_tensor*2-1)
        score = score.cpu().item()
        
        return score
    
    def calculate_ssim(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        """计算单张图像的SSIM
        
        Args:
            img_pred: 预测图像
            img_gt: 真实图像
            mask_pred: 预测图像的蒙版
            mask_gt: 真实图像的蒙版
            
        Returns:
            SSIM分数
        """
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32) / 255
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32) / 255
            img_gt = img_gt * mask_gt
            
        img_pred_tensor = torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
            
        score = self.ssim_metric_calculator(img_pred_tensor, img_gt_tensor)
        score = score.cpu().item()
        
        return score
    
    def calculate_video_lpips(self, video_pred, video_gt, mask_pred=None, mask_gt=None):
        """计算视频的LPIPS
        
        Args:
            video_pred: 预测视频，列表形式，每个元素为一帧图像
            video_gt: 真实视频，列表形式，每个元素为一帧图像
            mask_pred: 预测视频的蒙版，列表形式，可为None
            mask_gt: 真实视频的蒙版，列表形式，可为None
            
        Returns:
            平均LPIPS分数和每一帧的LPIPS分数列表
        """
        assert len(video_pred) == len(video_gt), "Videos should have the same number of frames."
        
        lpips_scores = []
        
        for i in tqdm(range(len(video_pred)), desc="Calculating LPIPS for video", disable=not SHOW_INNER_PROGRESS):
            frame_pred = video_pred[i]
            frame_gt = video_gt[i]
            
            frame_mask_pred = None if mask_pred is None else mask_pred[i]
            frame_mask_gt = None if mask_gt is None else mask_gt[i]
            
            lpips_score = self.calculate_lpips(frame_pred, frame_gt, frame_mask_pred, frame_mask_gt)
            lpips_scores.append(lpips_score)
        
        # 计算平均LPIPS分数
        avg_lpips = np.mean(lpips_scores)
        
        return avg_lpips, lpips_scores
    
    def calculate_video_ssim(self, video_pred, video_gt, mask_pred=None, mask_gt=None):
        """计算视频的SSIM
        
        Args:
            video_pred: 预测视频，列表形式，每个元素为一帧图像
            video_gt: 真实视频，列表形式，每个元素为一帧图像
            mask_pred: 预测视频的蒙版，列表形式，可为None
            mask_gt: 真实视频的蒙版，列表形式，可为None
            
        Returns:
            平均SSIM分数和每一帧的SSIM分数列表
        """
        assert len(video_pred) == len(video_gt), "Videos should have the same number of frames."
        
        ssim_scores = []
        
        for i in tqdm(range(len(video_pred)), desc="Calculating SSIM for video", disable=not SHOW_INNER_PROGRESS):
            frame_pred = video_pred[i]
            frame_gt = video_gt[i]
            
            frame_mask_pred = None if mask_pred is None else mask_pred[i]
            frame_mask_gt = None if mask_gt is None else mask_gt[i]
            
            ssim_score = self.calculate_ssim(frame_pred, frame_gt, frame_mask_pred, frame_mask_gt)
            ssim_scores.append(ssim_score)
        
        # 计算平均SSIM分数
        avg_ssim = np.mean(ssim_scores)
        
        return avg_ssim, ssim_scores
    
    def calculate_video_metrics(self, video_pred, video_gt, mask_pred=None, mask_gt=None):
        """计算视频的LPIPS, SSIM和姿态相似度指标
        
        Args:
            video_pred: 预测视频，列表形式，每个元素为一帧图像
            video_gt: 真实视频，列表形式，每个元素为一帧图像
            mask_pred: 预测视频的蒙版，列表形式，可为None
            mask_gt: 真实视频的蒙版，列表形式，可为None
            
        Returns:
            包含LPIPS、SSIM、姿态相似度平均分数以及每帧分数的字典
        """
        # 计算LPIPS指标
        avg_lpips, lpips_scores = self.calculate_video_lpips(video_pred, video_gt, mask_pred, mask_gt)
        
        # 计算SSIM指标
        avg_ssim, ssim_scores = self.calculate_video_ssim(video_pred, video_gt, mask_pred, mask_gt)
        
        # 计算姿态相似度指标（如果加载了姿态模型）
        avg_pose_l2_dist = None
        pose_l2_distances = None
        
        if self.pose_model is not None:
            avg_pose_l2_dist, pose_l2_distances = self.calculate_video_pose_similarity(video_pred, video_gt)
        
        # 返回结果字典
        results = {
            "avg_lpips": avg_lpips,
            "lpips_per_frame": lpips_scores,
            "avg_ssim": avg_ssim,
            "ssim_per_frame": ssim_scores
        }
        
        # 如果计算了姿态相似度，添加到结果中
        if avg_pose_l2_dist is not None:
            results["avg_pose_l2_dist"] = avg_pose_l2_dist
            results["pose_l2_dist_per_frame"] = pose_l2_distances
        
        return results
    
    def build_all_id_features_database(self, ref_imgs_dir: str, cache_file: str = None, 
                                   force_rebuild: bool = False, ref_types: List[str] = None):
        """构建所有参考图像的ID特征数据库
        
        Args:
            ref_imgs_dir: 参考图像目录
            cache_file: 缓存文件路径，如果为None则自动生成
            force_rebuild: 是否强制重建，即使缓存存在
            ref_types: 要包含的ref类型列表，如['sim', 'mid', 'diff']，None表示使用旧格式
        
        Returns:
            all_id_features: numpy数组 (N, feature_dim)
            ref_img_names: 参考图像名称列表
        """
        # 如果未指定ref_types，使用旧的命名规则
        if ref_types is None:
            ref_types = ['ref']  # 旧格式：xxxxx_ref.png
        
        if cache_file is None:
            cache_suffix = '_'.join(ref_types)
            cache_file = os.path.join(ref_imgs_dir, f"all_id_features_{cache_suffix}.npz")
        
        # 检查缓存
        if not force_rebuild and os.path.exists(cache_file):
            try:
                print(f"Loading cached ID features from {cache_file}")
                data = np.load(cache_file, allow_pickle=True)
                all_id_features = data['features']
                ref_img_names = data['names'].tolist()
                print(f"Loaded {len(ref_img_names)} cached ID features")
                return all_id_features, ref_img_names
            except Exception as e:
                print(f"Error loading cache: {e}, rebuilding...")
        
        if self.face_model is None:
            raise ValueError("Face model not loaded. Call load_face_model first.")
        
        print(f"Building ID features database from {ref_imgs_dir} for types: {ref_types}")
        
        # 获取所有参考图像文件
        ref_img_files = []
        for ref_type in ref_types:
            for ext in ['png', 'jpg', 'jpeg']:
                pattern = os.path.join(ref_imgs_dir, f"*_{ref_type}.{ext}")
                matched_files = glob.glob(pattern)
                ref_img_files.extend(matched_files)
                print(f"Found {len(matched_files)} files matching pattern: {pattern}")
        
        if not ref_img_files:
            raise ValueError(f"No reference images found in {ref_imgs_dir} for types {ref_types}")
        
        ref_img_files = list(set(ref_img_files))  # 去重
        ref_img_files.sort()  # 确保顺序一致
        
        all_id_features = []
        ref_img_names = []
        
        print(f"Processing {len(ref_img_files)} reference images...")
        for img_path in tqdm(ref_img_files, desc="Extracting ID features", disable=not SHOW_INNER_PROGRESS):
            try:
                # 加载图像
                img = Image.open(img_path)
                
                # 检测并对齐人脸
                aligned_face = self.detect_and_align_face(img)
                if aligned_face is None:
                    print(f"Warning: No face detected in {img_path}, skipping")
                    continue
                
                # 提取ID特征
                id_feature = self.extract_face_feature(aligned_face, use_flip=True)
                
                all_id_features.append(id_feature.numpy())
                ref_img_names.append(os.path.basename(img_path))
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if not all_id_features:
            raise ValueError("No valid ID features extracted from reference images")
        
        # 转换为numpy数组
        all_id_features = np.stack(all_id_features, axis=0)
        
        # 保存缓存
        try:
            np.savez(cache_file, features=all_id_features, names=np.array(ref_img_names))
            print(f"Saved ID features database to {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
        
        print(f"Built ID features database: {all_id_features.shape[0]} features, dim={all_id_features.shape[1]}")
        
        return all_id_features, ref_img_names
    
    def calculate_id_retrieval_metrics(self, video_frames: List, all_id_features: np.ndarray, ref_img_names: List[str], target_ref_name: str = None):
        """计算ID检索指标
        
        Args:
            video_frames: 视频帧列表
            all_id_features: 所有参考图像的ID特征数据库 (N, feature_dim)
            ref_img_names: 参考图像名称列表
            target_ref_name: 目标参考图像名称（用于计算准确率）
        
        Returns:
            retrieval_results: 字典包含top1、top5、top10准确率和详细结果
        """
        if self.face_model is None:
            raise ValueError("Face model not loaded. Call load_face_model first.")
        
        if len(all_id_features) != len(ref_img_names):
            raise ValueError("all_id_features and ref_img_names length mismatch")
        
        # 提取视频帧的ID特征
        frame_id_features = []
        valid_frame_indices = []
        
        for i, frame in enumerate(tqdm(video_frames, desc="Extracting frame ID features", disable=not SHOW_INNER_PROGRESS)):
            try:
                # 转换为PIL Image（如果是numpy数组）
                if isinstance(frame, np.ndarray):
                    frame = Image.fromarray(frame)
                
                # 检测并对齐人脸
                aligned_face = self.detect_and_align_face(frame)
                if aligned_face is None:
                    print(f"Warning: No face detected in frame {i}, skipping")
                    continue
                
                # 提取ID特征
                id_feature = self.extract_face_feature(aligned_face, use_flip=True)
                frame_id_features.append(id_feature.numpy())
                valid_frame_indices.append(i)
                
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                continue
        
        if not frame_id_features:
            print("Warning: No valid frames for ID retrieval")
            return {
                'top1_accuracy': 0.0,
                'top5_accuracy': 0.0,
                'top10_accuracy': 0.0,
                'valid_frames_count': 0,
                'total_frames_count': len(video_frames),
                'frame_results': []
            }
        
        frame_id_features = np.stack(frame_id_features, axis=0)  # (M, feature_dim)
        
        # 计算余弦相似度矩阵
        # frame_id_features: (M, feature_dim)
        # all_id_features: (N, feature_dim)
        # similarities: (M, N)
        frame_id_features_norm = frame_id_features / (np.linalg.norm(frame_id_features, axis=1, keepdims=True) + 1e-8)
        all_id_features_norm = all_id_features / (np.linalg.norm(all_id_features, axis=1, keepdims=True) + 1e-8)
        
        similarities = np.dot(frame_id_features_norm, all_id_features_norm.T)  # (M, N)
        
        # 对每一帧获取top-k结果
        top1_matches = []
        top5_matches = []
        top10_matches = []
        frame_results = []
        
        for i, sim_scores in enumerate(similarities):
            # 获取按相似度降序排序的索引
            sorted_indices = np.argsort(sim_scores)[::-1]
            
            # 获取top-k结果
            top1_idx = sorted_indices[0]
            top5_indices = sorted_indices[:5]
            top10_indices = sorted_indices[:10]
            
            # 记录结果
            frame_result = {
                'frame_idx': valid_frame_indices[i],
                'top1_name': ref_img_names[top1_idx],
                'top1_score': float(sim_scores[top1_idx]),
                'top5_names': [ref_img_names[idx] for idx in top5_indices],
                'top5_scores': [float(sim_scores[idx]) for idx in top5_indices],
                'top10_names': [ref_img_names[idx] for idx in top10_indices],
                'top10_scores': [float(sim_scores[idx]) for idx in top10_indices],
            }
            frame_results.append(frame_result)
            
            # 如果提供了目标参考名称，计算准确率
            if target_ref_name is not None:
                top1_match = ref_img_names[top1_idx] == target_ref_name
                top5_match = target_ref_name in [ref_img_names[idx] for idx in top5_indices]
                top10_match = target_ref_name in [ref_img_names[idx] for idx in top10_indices]
                
                top1_matches.append(top1_match)
                top5_matches.append(top5_match)
                top10_matches.append(top10_match)
        
        # 计算整体准确率
        if target_ref_name is not None:
            top1_accuracy = np.mean(top1_matches) if top1_matches else 0.0
            top5_accuracy = np.mean(top5_matches) if top5_matches else 0.0
            top10_accuracy = np.mean(top10_matches) if top10_matches else 0.0
        else:
            top1_accuracy = None
            top5_accuracy = None
            top10_accuracy = None
        
        return {
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'top10_accuracy': top10_accuracy,
            'valid_frames_count': len(frame_id_features),
            'total_frames_count': len(video_frames),
            'target_ref_name': target_ref_name,
            'frame_results': frame_results
        }
    
    def convert_frames_to_tensors(self, frames, normalize_range='0_1'):
        """
        Convert frame list to torch tensors for warping error calculation
        
        Args:
            frames: list of frames (PIL Images or numpy arrays)
            normalize_range: '0_1' for [0,1] range, '-1_1' for [-1,1] range
            
        Returns:
            list of torch tensors, each (H,W,3), float32 in specified range
        """
        tensor_frames = []
        
        for frame in frames:
            # Convert to numpy if PIL Image
            if isinstance(frame, Image.Image):
                frame_np = np.array(frame, dtype=np.float32)
            elif isinstance(frame, np.ndarray):
                frame_np = frame.astype(np.float32)
            else:
                raise ValueError(f"Unsupported frame type: {type(frame)}")
            
            # Ensure RGB format and normalize
            if frame_np.max() > 1.0:  # Assume [0, 255] range
                frame_np = frame_np / 255.0
            
            # Convert to specified range
            if normalize_range == '-1_1':
                frame_np = frame_np * 2.0 - 1.0
            elif normalize_range != '0_1':
                raise ValueError(f"Unsupported normalize_range: {normalize_range}")
            
            # Convert to torch tensor
            frame_tensor = torch.from_numpy(frame_np)
            tensor_frames.append(frame_tensor)
            
        return tensor_frames
    
    def calculate_warping_error(self, orig_frames, gen_frames, face_masks, 
                              fb_thresh=1.0, weights_path=None, 
                              video_length=None, random_seed=None):
        """
        Calculate warping error between original and generated video frames
        
        Args:
            orig_frames: list of original frames (PIL Images or numpy arrays)
            gen_frames: list of generated frames (PIL Images or numpy arrays)
            face_masks: list of face mask frames (PIL Images or numpy arrays), binary {0,1}
            fb_thresh: float, forward-backward consistency threshold (pixels)
            weights_path: str or None, path to local RAFT weights (.pth)
            video_length: int or None, if specified, randomly crop video to this length
            random_seed: int or None, random seed for reproducible cropping
            
        Returns:
            dict with warping error metrics
        """
        try:
            # Convert frames to torch tensors in [0,1] range
            orig_tensor_frames = self.convert_frames_to_tensors(orig_frames, '0_1')
            gen_tensor_frames = self.convert_frames_to_tensors(gen_frames, '0_1')
            mask_tensor_frames = self.convert_frames_to_tensors(face_masks, '0_1')
            
            # Ensure all frame lists have same length
            assert len(orig_tensor_frames) == len(gen_tensor_frames) == len(mask_tensor_frames), \
                "All frame lists must have the same length"
            
            # Calculate warping error using imported function
            warping_result = compute_ref_warping_error(
                orig_frames=orig_tensor_frames,
                gen_frames=gen_tensor_frames,
                face_masks=mask_tensor_frames,
                device=str(self.device),
                fb_thresh=fb_thresh,
                weights_path=weights_path,
                video_length=video_length,
                random_seed=random_seed
            )
            
            return warping_result
            
        except Exception as e:
            print(f"Error calculating warping error: {e}")
            return {
                "short_list": [],
                "long_list": [],
                "mean_short": float("nan"),
                "mean_long": float("nan"),
                "mean_total": float("nan")
            }
