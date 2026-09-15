import sys
sys.path.append("./")

import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import tyro
import ray
import torch
from PIL import Image
from tqdm import tqdm
import os, json, re, tempfile, stat
from decord import VideoReader, cpu
import multiprocessing as mp
import cv2
import torch.nn as nn
import torchvision
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF


from eval_tools.metrics_calculator_facebench import MetricsCalculator
from swapface_benchmark.roi import read_face_boxes, scaled_box

DECODE_THREADS = max(1, int(os.environ.get('BENCHMARK_DECODE_THREADS', '2')))

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv')
PACKAGE_ROOT = str(Path(__file__).resolve().parents[2])

os.environ['RAY_DEDUP_LOGS'] = '0'

# os.environ['RUN_MODE'] = 'debug'

@dataclass
class EvalConfig:
    """评估配置类"""
    mapping: str = os.getenv("FACEBENCH_MAPPING", "")
    project_dir: str = PACKAGE_ROOT
    # source_video_dir: str = project_dir + "/eval_datas/FaceBench/merge_data"
    source_video_dir: str = os.getenv("SOURCE_VIDEO_DIR", project_dir + "/eval_datas/FaceBench/merge_data_1920")
    mask_video_dir: str = None  # 不再需要单独的mask目录
    ref_imgs_dir: str = None  # 不再需要单独的ref目录

    # 输出参数
    # method_name: str = "simswap"
    method_name: str = os.getenv("METHOD_NAME", "")
    output_dir: str = os.getenv("OUTPUT_DIR", "")
    save_detailed: bool = True  # 是否保存详细结果
    # target_video_dir: str = f"data/FaceForensicspp_mutli_method_result/output_{method_name}"
    # target_video_dir: str = f"eval_datas/FaceBench/inswapper_results"
    # target_video_dir: str = f"/mnt/nas/share/home/zmz/code/face/SimSwap/output_simswap_facebench"
    target_video_dir: str = os.getenv("TARGET_VIDEO_DIR", "")

    # 模型路径
    face_model_path: str = os.getenv("FACE_MODEL_PATH", project_dir + "/eval_tools/third_party/CosFace_pytorch/checkpoints/ACC99.28.pth")  # 人脸模型路径
    face_detect_model_path: str = os.getenv("FACE_DETECT_MODEL_PATH", project_dir + "/eval_tools/third_party/insightface_func/checkpoints")  # 人脸检测模型路径
    pose_model_path: str = os.getenv("POSE_MODEL_PATH", project_dir + "/eval_tools/third_party/deep_head_pose/checkpoints/hopenet_robust_alpha1.pkl")
    gaze_model_path: str = os.getenv("GAZE_MODEL_PATH", project_dir + "/models/L2CSNet_gaze360.pkl")
    
    # Deep3DFaceRecon模型路径
    deep3d_checkpoints_dir: str = os.getenv("DEEP3D_CHECKPOINTS_DIR", project_dir + "/eval_tools/third_party/Deep3DFaceRecon_pytorch/checkpoints")  # Deep3D模型checkpoints目录
    deep3d_bfm_folder: str = os.getenv("DEEP3D_BFM_FOLDER", project_dir + "/eval_tools/third_party/Deep3DFaceRecon_pytorch/BFM")  # BFM模型文件夹

    # 评估参数
    # max_frames: Optional[int] = None  # 最大帧数
    # max_frames: Optional[int] = 10  # 最大帧数
    max_frames: Optional[int] = int(os.getenv("MAX_EVAL_FRAMES", "0")) or None  # None 表示全测
    frame_stride: int = int(os.getenv("FRAME_STRIDE", "1"))
    video_num: Optional[int] = None  # 最多处理的视频数量，None表示处理所有视频
    use_align: bool = False  # 是否使用对齐数据
    target_face_type: str = "png"  # 目标人脸图像类型
    random_sampling: bool = os.getenv("RANDOM_SAMPLING", "0") == "1"
    
    # 测试选项配置
    # enable_face_sim: bool = True   # 是否测试人脸相似度
    # enable_lpips: bool = True  # 是否测试LPIPS
    # enable_ssim: bool = True  # 是否测试SSIM
    # enable_pose: bool = True  # 是否测试姿态距离
    # enable_gaze: bool = True  # 是否测试视线距离
    # enable_exp_gamma: bool = True  # 是否测试expression和gamma距离
    # enable_id_retrieval: bool = True  # 是否测试ID检索
    # enable_warping_error: bool = True  # 是否测试warping error

    enable_face_sim: bool = True   # 是否测试人脸相似度
    enable_lpips: bool = False  # 是否测试LPIPS
    enable_ssim: bool = False  # 是否测试SSIM
    enable_pose: bool = True  # 是否测试姿态距离
    enable_gaze: bool = True  # 是否测试视线距离
    enable_exp_gamma: bool = True  # 是否测试expression和gamma距离
    enable_id_retrieval: bool = False  # 是否测试ID检索
    enable_warping_error: bool = False  # 是否测试warping error
    
    # Warping error 参数
    warping_fb_thresh: float = 1.0  # forward-backward consistency threshold (pixels)
    warping_weights_path: Optional[str] = None  # RAFT模型权重路径
    warping_video_length: Optional[int] = 11  # 随机截取的视频长度
    warping_random_seed: Optional[int] = 42  # 随机种子

    # 并行处理参数
    num_gpus: int = int(os.getenv("NUM_GPUS", "4"))  # GPU数量
    batch_size: int = 1  # 每个worker处理的视频数量
    
    # Resume参数
    enable_resume: bool = False  # 是否启用断点续传
    override_existing: bool = False  # 是否覆盖已存在的结果文件
    single_result_dir: str = "single_results"  # 单个视频结果存储目录

    # # 输出参数自动依赖于 method_name
    # @property
    # def output_dir(self) -> str:
    #     return f"output_lzk/cvpr/eval_facebench/{self.method_name}"

def un_norm_clip(x1):
    x = x1*1.0 # to avoid changing the original tensor or clone() can be used
    reduce=False
    if len(x.shape)==3:
        x = x.unsqueeze(0)
        reduce=True
    x[:,0,:,:] = x[:,0,:,:] * 0.26862954 + 0.48145466
    x[:,1,:,:] = x[:,1,:,:] * 0.26130258 + 0.4578275
    x[:,2,:,:] = x[:,2,:,:] * 0.27577711 + 0.40821073
    
    if reduce:
        x = x.squeeze(0)
    return x


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)


# def find_video_pairs(input_dir: str, mask_root_dir: Optional[str] = None, 
#                     ref_imgs_dir: Optional[str] = None,
#                     target_face_type: str = "png", 
#                     use_align: bool = False,
#                     video_num: Optional[int] = None) -> List[Tuple[str, str, str, str, str]]:
#     """
#     查找视频对，包括输入视频、掩码视频、目标人脸等
    
#     Args:
#         input_dir: 源视频目录
#         mask_root_dir: 掩码视频目录
#         ref_imgs_dir: 参考图像目录
#         target_face_type: 参考图像文件类型
#         use_align: 是否使用对齐数据
#         video_num: 最多返回的视频对数量，None表示返回所有找到的视频对
    
#     Returns:
#         List of (input_path, mask_path, target_path, rel_path, rel_path_no_ext)
#     """
#     pairs = []
#     for root, _, files in os.walk(input_dir):
#         for file in files:
#             if file.lower().endswith(VIDEO_EXTENSIONS):
#                 input_path = os.path.join(root, file)

#                 # 跳过已经有后缀的文件
#                 if "_pose" in file or "_target" in file or "_mask" in file or "_swapped" in file:
#                     continue

#                 if use_align and "aligned" not in file:
#                     continue
#                 elif not use_align and "aligned" in file:
#                     continue
                
#                 # 获取基础文件名（不含扩展名）
#                 base_name = os.path.splitext(file)[0]
#                 rel_path = os.path.relpath(input_path, input_dir)
#                 rel_path_no_ext, _ = os.path.splitext(rel_path)

#                 # 构建掩码路径 - 在mask_root_dir中查找 base_name + "_mask.mp4"
#                 if mask_root_dir is None:
#                     mask_path = os.path.join(os.path.dirname(input_path), base_name + "_mask.mp4")
#                 else:
#                     mask_path = os.path.join(mask_root_dir, base_name + "_mask.mp4")

#                 # 构建参考图像路径 - 在ref_imgs_dir中查找 base_name + "_ref.png"
#                 if ref_imgs_dir is None:
#                     target_path = os.path.join(os.path.dirname(input_path), base_name + "_ref." + target_face_type)
#                 else:
#                     target_path = os.path.join(ref_imgs_dir, base_name + "_ref." + target_face_type)

#                 if os.path.exists(input_path) and os.path.exists(mask_path) and os.path.exists(target_path):
#                     pairs.append((input_path, mask_path, target_path, rel_path, rel_path_no_ext))
                    
#                     # 如果设置了视频数量限制且达到限制，提前返回
#                     if video_num is not None and len(pairs) >= video_num:
#                         print(f"已找到 {video_num} 个视频对，达到限制数量，停止搜索")
#                         return pairs
#                 else:
#                     if not os.path.exists(mask_path):
#                         print(f"Warning: Mask file not found for {input_path} at {mask_path}")
#                     if not os.path.exists(target_path):
#                         print(f"Warning: Target file not found for {input_path} at {target_path}")
                        
#     return pairs

def find_video_pairs_new(input_dir: str, 
                         use_align: bool = False,
                         video_num: Optional[int] = None) -> List[Tuple[str, str, dict, str, str]]:
    """
    查找视频对，支持新的目录结构
    
    Args:
        input_dir: 源视频目录（包含source、mask、ref）
        use_align: 是否使用对齐数据
        video_num: 最多返回的视频对数量，None表示返回所有找到的视频对
    
    Returns:
        List of (source_path, mask_path, ref_paths_dict, video_id, video_id)
        其中 ref_paths_dict = {'sim': path, 'mid': path, 'diff': path}
    """
    pairs = []
    processed_videos = set()  # 防止重复处理
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(VIDEO_EXTENSIONS):
                continue
            
            # 只处理源视频（5位数字，无下划线后缀）
            if "_" in file:
                continue
            
            if use_align and "aligned" not in file:
                continue
            elif not use_align and "aligned" in file:
                continue
            
            # 获取基础文件名（5位数字）
            base_name = os.path.splitext(file)[0]
            
            # 避免重复处理
            if base_name in processed_videos:
                continue
            
            source_path = os.path.join(root, file)
            
            # 构建mask路径
            boxes_path = os.path.join(root, base_name + "_boxes.json")
            mask_path = boxes_path if os.path.isfile(boxes_path) else os.path.join(root, base_name + "_mask.mp4")
            
            # 构建三个ref路径
            ref_paths = {
                'sim': os.path.join(root, base_name + "_ref_sim.png"),
                'mid': os.path.join(root, base_name + "_ref_mid.png"),
                'diff': os.path.join(root, base_name + "_ref_diff.png")
            }
            
            # 检查所有必需文件是否存在
            if not os.path.exists(source_path):
                continue
            if not os.path.exists(mask_path):
                print(f"Warning: Mask file not found for {source_path}")
                continue
            
            # 检查至少一个ref图像存在
            valid_refs = {k: v for k, v in ref_paths.items() if os.path.exists(v)}
            if not valid_refs:
                print(f"Warning: No reference images found for {source_path}")
                continue
            
            # 只保留存在的ref
            ref_paths = valid_refs
            
            video_id = base_name
            pairs.append((source_path, mask_path, ref_paths, video_id, video_id))
            processed_videos.add(base_name)
            
            # 如果设置了视频数量限制且达到限制，提前返回
            if video_num is not None and len(pairs) >= video_num:
                print(f"已找到 {video_num} 个视频对，达到限制数量，停止搜索")
                return pairs
    
    return pairs


# def get_corresponding_target_videos(source_pairs: List[Tuple], target_video_dir: str) -> List[str]:
#     """获取对应的目标视频路径"""
#     target_videos = []
    
#     for pair in source_pairs:
#         source_video_path = pair[0]  # source video path
#         base_name = os.path.splitext(os.path.basename(source_video_path))[0]
        
#         # 在目标目录中查找 base_name + "_swapped.mp4"
#         target_video_name = base_name + "_swapped.mp4"
#         target_video_path = os.path.join(target_video_dir, target_video_name)
        
#         if os.path.exists(target_video_path):
#             target_videos.append(target_video_path)
#         else:
#             print(f"Warning: Target video not found for {source_video_path} at {target_video_path}")
#             target_videos.append(None)
    
#     return target_videos


def get_corresponding_target_videos_new(source_pairs: List[Tuple], 
                                       target_video_dir: str) -> List[dict]:
    """
    获取对应的目标视频路径（支持sim/mid/diff三个子目录）
    
    Returns:
        List of dict: {'sim': path, 'mid': path, 'diff': path} 或 None
    """
    target_videos_list = []
    
    for pair in source_pairs:
        video_id = pair[3]  # base_name (5位数字)
        ref_paths = pair[2]  # ref_paths_dict
        
        target_videos = {}
        
        # 对每个ref类型查找对应的目标视频
        for ref_type in ref_paths.keys():
            target_subdir = os.path.join(target_video_dir, ref_type)
            target_video_name = video_id + "_swapped.mp4"
            target_video_path = os.path.join(target_subdir, target_video_name)
            
            if os.path.exists(target_video_path):
                target_videos[ref_type] = target_video_path
            else:
                print(f"Warning: Target video not found: {target_video_path}")
        
        if target_videos:
            target_videos_list.append(target_videos)
        else:
            target_videos_list.append(None)
    
    return target_videos_list



def extract_video_frames(video_path: str, frame_indices: List[int], reader=None) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
    # 优先 decord
    try:
        from decord import VideoReader, cpu
        vr = reader if reader is not None else VideoReader(video_path, ctx=cpu(0), num_threads=DECODE_THREADS)
        frames = vr.get_batch(frame_indices).asnumpy()  # [N,H,W,C]
        ts = np.array([vr.get_frame_timestamp(i) for i in frame_indices], dtype=np.float32)
        return list(frames), ts, frame_indices
    except Exception as e_decord:
        # 兜底：用 OpenCV
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Both decord and cv2 fail to open: {video_path} | decord_err={e_decord}")
        frames, ts = [], []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for i in frame_indices:
            if i >= total: break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            ts.append(i / fps)
        cap.release()
        if not frames:
            raise RuntimeError(f"cv2 fallback also failed: {video_path}")
        return frames, np.array(ts, dtype=np.float32), frame_indices[:len(frames)]

def calculate_bbox_from_mask(mask_frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """从掩码帧计算边界框"""
    if len(mask_frame.shape) == 3:
        mask = cv2.cvtColor(mask_frame, cv2.COLOR_RGB2GRAY)
    else:
        mask = mask_frame
    
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    
    return (x, y, x + w, y + h)


def expand_bbox(bbox: Optional[Tuple[int, int, int, int]], 
               frame_shape: Tuple[int, int], 
               expand_ratio: Tuple[float, float, float, float] = (0.5, 0.5, 0.75, 0.25)) -> Optional[Tuple[int, int, int, int]]:
    """扩展边界框"""
    if bbox is None:
        return None
    
    x_min, y_min, x_max, y_max = bbox
    bbox_width = abs(x_max - x_min)
    bbox_height = abs(y_max - y_min)
    
    x_min -= expand_ratio[0] * bbox_width
    x_max += expand_ratio[1] * bbox_width
    y_min -= expand_ratio[2] * bbox_height
    y_max += expand_ratio[3] * bbox_height
    
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(frame_shape[1], x_max)
    y_max = min(frame_shape[0], y_max)
    
    return int(x_min), int(y_min), int(x_max), int(y_max)


def crop_face(frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """裁剪帧中的人脸区域"""
    if bbox is None:
        return frame
    
    x_min, y_min, x_max, y_max = bbox
    face = frame[y_min:y_max, x_min:x_max]
    return face


def compute_aligned_frame_ids(frame_timestamps: np.ndarray, origin_frame_count: int, target_frame_count: int) -> List[int]:
    """
    从原始视频帧时间戳中，提取等间隔的 target_frame_count 个帧对应的帧索引。

    参数：
        frame_timestamps: np.ndarray, shape [N, 2]
            每一帧的时间范围 (start_time, end_time)
        origin_frame_count: int
            原始视频总帧数（应该等于 len(frame_timestamps)）
        target_frame_count: int
            想要采样的目标帧数

    返回：
        frame_ids: list[int]
            在原始帧序列中应该采样的帧索引
    """
    assert origin_frame_count == len(frame_timestamps), "帧数不一致"
    duration = frame_timestamps[-1].mean()
    timestamps = np.linspace(0., duration, target_frame_count)
    frame_ids = np.argmax(
        np.logical_and(
            timestamps[:, None] >= frame_timestamps[None, :, 0],
            timestamps[:, None] <= frame_timestamps[None, :, 1]
        ),
        axis=1
    ).tolist()
    return frame_ids

def resize_frame_index(frame_idx: int, src_count: int, target_count: int) -> int:
    """Match training/inference resize: linspace downsample, last-frame pad."""
    if src_count <= 1 or target_count <= 1:
        return 0
    if src_count >= target_count:
        index = int(frame_idx * (src_count - 1) / (target_count - 1))
        return min(max(index, 0), src_count - 1)
    if frame_idx < src_count:
        return frame_idx
    return src_count - 1

def sanitize_video_id(s: str) -> str:
    # 仅保留常见安全字符，其余转为下划线，并去掉头尾空白
    s = s.strip()
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', s)
    return s or "unnamed"


@ray.remote(num_gpus=1)
class ComprehensiveEvaluator:
    """Ray远程类，用于并行计算综合评估指标"""
    
    def __init__(self, config: EvalConfig):
        import os
        # breakpoint()
        # Ray会自动分配CUDA设备，无需手动指定device
        self.config = config
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # 绑定到本地0号
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print("[Actor Device] =", self.device)
        self.metrics_calculator = MetricsCalculator(device=self.device)
        
        # 加载各种模型
        if (config.enable_face_sim or config.enable_id_retrieval) and config.face_model_path:
            self.metrics_calculator.load_face_model(config.face_model_path)
        
        if (config.enable_face_sim or config.enable_id_retrieval) and config.face_detect_model_path:
            self.metrics_calculator.load_face_detect_align_model(config.face_detect_model_path)
        
        if config.enable_pose and config.pose_model_path:
            self.metrics_calculator.load_pose_model(config.pose_model_path)
        
        if config.enable_gaze and config.gaze_model_path:
            self.metrics_calculator.load_gaze_model(config.gaze_model_path)
        
        if config.enable_exp_gamma and config.deep3d_checkpoints_dir:
            self.metrics_calculator.load_deep3d_model(config.deep3d_checkpoints_dir, config.deep3d_bfm_folder)
        
        # 构建ID特征数据库（如果启用ID检索）
        self.all_id_features_by_ref = {}  # 改为字典，按ref_type分别存储
        self.ref_img_names_by_ref = {}
        
        if config.enable_id_retrieval:
            try:
                print("Building ID retrieval databases for each ref type...")
                
                # 为每个ref类型分别构建数据库
                for ref_type in ['sim', 'mid', 'diff']:
                    try:
                        print(f"Building database for {ref_type}...")
                        features, names = self.metrics_calculator.build_all_id_features_database(
                            config.source_video_dir, 
                            force_rebuild=False,
                            ref_types=[ref_type]  # 只包含当前类型
                        )
                        self.all_id_features_by_ref[ref_type] = features
                        self.ref_img_names_by_ref[ref_type] = names
                        print(f"  {ref_type}: {len(names)} reference images")
                    except Exception as e:
                        print(f"Warning: Failed to build database for {ref_type}: {e}")
                
                if not self.all_id_features_by_ref:
                    raise ValueError("No ID features database built for any ref type")
                    
                print(f"ID retrieval databases ready for {len(self.all_id_features_by_ref)} ref types")
                
            except Exception as e:
                print(f"Warning: Failed to build ID retrieval databases: {e}")
                print("ID retrieval will be disabled for this session")
                config.enable_id_retrieval = False
            
        # 创建单个结果存储目录
        self.single_result_dir = os.path.abspath(os.path.join(config.output_dir, config.single_result_dir))
        os.makedirs(self.single_result_dir, exist_ok=True)
    
    def _get_result_file_path(self, video_id: str) -> str:
        """获取单个视频结果文件路径"""
        # 安全处理 video_id，避免 / \ 等特殊字符
        safe_video_id = video_id.replace("/", "_").replace("\\", "_").strip()
        return os.path.join(self.single_result_dir, f"{safe_video_id}.json")

    def _atomic_write_json(self, obj, dst_path: str):
        d = os.path.dirname(dst_path)
        os.makedirs(d, exist_ok=True)
        # 方案 1：mkstemp + replace
        try:
            fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
            with os.fdopen(fd, "w") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
                f.flush(); os.fsync(f.fileno())
            os.chmod(tmp, stat.S_IRUSR|stat.S_IWUSR|stat.S_IRGRP|stat.S_IWGRP|stat.S_IROTH)
            os.replace(tmp, dst_path)
            return
        except PermissionError:
            # 方案 2：.partial + fsync + replace（避免 mkstemp）
            tmp = os.path.join(d, os.path.basename(dst_path) + ".partial")
            try:
                with open(tmp, "w") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=2)
                    f.flush(); os.fsync(f.fileno())
                os.chmod(tmp, stat.S_IRUSR|stat.S_IWUSR|stat.S_IRGRP|stat.S_IWGRP|stat.S_IROTH)
                os.replace(tmp, dst_path)
                return
            except PermissionError:
                # 方案 3：直接写最终文件（非原子兜底）
                with open(dst_path, "w") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=2)

    def _save_single_result(self, result: dict):
        video_id = result['video_id']
        result_file = self._get_result_file_path(video_id)

        # numpy -> 可序列化
        serializable_result = {}
        for k, v in result.items():
            try:
                import numpy as np
                if isinstance(v, np.ndarray):
                    serializable_result[k] = v.tolist()
                elif isinstance(v, np.floating):
                    serializable_result[k] = float(v)
                elif isinstance(v, list) and v and isinstance(v[0], np.floating):
                    serializable_result[k] = [float(x) for x in v]
                else:
                    serializable_result[k] = v
            except Exception:
                serializable_result[k] = v

        # 写入（带重试，防 NFS 抖动）
        tries, last_err = 3, None
        for _ in range(tries):
            try:
                self._atomic_write_json(serializable_result, result_file)
                print(f"Saved result for video: {repr(video_id)} -> {result_file}")
                return
            except PermissionError as e:
                last_err = e
                import time; time.sleep(0.2)
        raise last_err
    
    def _load_single_result(self, video_id: str) -> Optional[dict]:
        """加载单个视频评估结果"""
        result_file = self._get_result_file_path(video_id)
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading result for {video_id}: {e}")
                return None
        return None
    
    def _should_skip_video(self, video_id: str) -> bool:
        """判断是否应该跳过这个视频的评估"""
        if not self.config.enable_resume:
            return False
        
        result_file = self._get_result_file_path(video_id)
        if os.path.exists(result_file):
            if self.config.override_existing:
                print(f"Override enabled, re-evaluating video: {video_id}")
                return False
            else:
                print(f"Result exists, skipping video: {video_id}")
                return True
        return False
    
    def evaluate_video_batch(self, video_batch_data: List[dict]) -> List[dict]:
        """评估一批视频的综合指标"""
        results = []
        
        for video_data in video_batch_data:
            video_id = video_data.get('video_id', 'unknown')
            
            # 检查是否应该跳过这个视频
            if self._should_skip_video(video_id):
                # 加载已存在的结果
                existing_result = self._load_single_result(video_id)
                if existing_result:
                    results.append(existing_result)
                    continue
            
            try:
                result = self._evaluate_single_video(video_data)
                # 立即保存单个结果
                self._save_single_result(result)
                results.append(result)
            except Exception as e:
                print(f"Error evaluating video {video_data.get('video_id', 'unknown')}: {str(e)}")
                error_result = {
                    'video_id': video_data.get('video_id', 'unknown'),
                    'error': str(e),
                    'ref_type': video_data.get('ref_type', 'sim'),
                    'face_sim_scores': [],
                    'avg_face_sim': None,
                    'face_sim_contrast_scores': [],
                    'avg_face_sim_contrast': None,
                    'lpips_scores': [],
                    'avg_lpips': None,
                    'ssim_scores': [],
                    'avg_ssim': None,
                    'pose_l2_distances': [],
                    'avg_pose_l2_dist': None,
                    'gaze_l2_distances': [],
                    'avg_gaze_l2_dist': None,
                    'gaze_cosine_similarities': [],
                    'avg_gaze_cosine_sim': None,
                    'frame_indices': []
                }
                # 保存错误结果
                self._save_single_result(error_result)
                results.append(error_result)
        
        return results
    
    def _evaluate_single_video(self, video_data: dict) -> dict:
        """评估单个视频的综合指标"""
        # breakpoint()
        source_video_path = video_data['source_video_path']
        target_video_path = video_data['target_video_path']
        mask_video_path = video_data['mask_video_path']
        ref_face_path = video_data['ref_face_path']
        video_id = video_data['video_id']
        max_frames = video_data.get('max_frames', None)
        if max_frames is not None and max_frames <= 0:
            max_frames = None
        random_sampling = video_data.get('random_sampling', False)
        frame_stride = int(video_data.get('frame_stride', 1))
        if frame_stride <= 0:
            raise ValueError(f"frame_stride must be positive, got {frame_stride}")

        # 获取当前视频的ref_type
        ref_type = video_data.get('ref_type', 'sim')  # 默认sim
        base_video_id = video_data.get('base_video_id', video_id.split('_')[0])  # 提取基础ID
        
        print(f"Evaluating video: {video_id}")


        # The generated video defines the evaluation timeline. The optional
        # linspace_trim_padding mode is used by experiments whose inference
        # uniformly downsamples long inputs and last-frame-pads short inputs.
        source_vr = VideoReader(source_video_path, ctx=cpu(0), num_threads=DECODE_THREADS)
        target_vr = VideoReader(target_video_path, ctx=cpu(0), num_threads=DECODE_THREADS)
        source_total_frames = len(source_vr)
        target_total_frames = len(target_vr)
        target_fps = float(target_vr.get_avg_fps())
        source_fps = float(source_vr.get_avg_fps())
        alignment_mode = os.getenv("FACEBENCH_FRAME_ALIGNMENT", "timestamp_strict")
        mask_vr = None
        mask_total_frames = None
        mask_fps = None
        boxes = read_face_boxes(mask_video_path) if mask_video_path and Path(mask_video_path).suffix.lower() == ".json" else None
        if boxes is not None and (self.config.enable_lpips or self.config.enable_ssim or self.config.enable_warping_error):
            raise ValueError("pixel-mask metrics require an explicit mask video; boxes are not segmentation")
        if mask_video_path and os.path.exists(mask_video_path) and boxes is None:
            mask_vr = VideoReader(mask_video_path, ctx=cpu(0), num_threads=DECODE_THREADS)
            mask_total_frames = len(mask_vr)
            mask_fps = float(mask_vr.get_avg_fps())
        if alignment_mode == "linspace_trim_padding":
            # Drop generated tail frames that only exist because a short input
            # was padded. For long inputs, reproduce inference's linspace map.
            dependency_lengths = [source_total_frames]
            if mask_total_frames is not None:
                dependency_lengths.append(mask_total_frames)
            eval_frame_count = min(target_total_frames, *dependency_lengths)
        else:
            eval_frame_count = target_total_frames
        candidate_frame_indices = np.arange(0, eval_frame_count, frame_stride)
        if max_frames is not None and max_frames < len(candidate_frame_indices):
            if random_sampling:
                np.random.seed(42)
                eval_frame_indices = np.random.choice(candidate_frame_indices, max_frames, replace=False)
                eval_frame_indices = np.sort(eval_frame_indices)
            else:
                eval_frame_indices = candidate_frame_indices[:max_frames]
        else:
            eval_frame_indices = candidate_frame_indices

        target_frame_indices = eval_frame_indices.tolist()
        if alignment_mode == "linspace_trim_padding":
            source_frame_indices_for_eval = [
                resize_frame_index(int(i), source_total_frames, target_total_frames)
                for i in eval_frame_indices
            ]
        else:
            source_frame_indices_for_eval = [int(round(int(i) / target_fps * source_fps)) for i in eval_frame_indices]
        if source_frame_indices_for_eval and max(source_frame_indices_for_eval) >= source_total_frames:
            raise RuntimeError(
                f"origin video is shorter than generated timeline: need source frame "
                f"{max(source_frame_indices_for_eval)}, available={source_total_frames}"
            )

        
        if boxes is not None and max(source_frame_indices_for_eval, default=-1) >= len(boxes):
            raise ValueError("face boxes do not cover the evaluated origin-video frames")

        # 提取目标视频帧
        target_frames, target_timestamps, target_frame_indices = extract_video_frames(
            target_video_path, target_frame_indices, reader=target_vr
        )
        target_frames = target_frames
        target_timestamps = target_timestamps
        target_frame_indices = target_frame_indices
        
        # 提取源视频帧
        source_frames, source_timestamps, source_frame_indices = extract_video_frames(
            source_video_path, source_frame_indices_for_eval, reader=source_vr
        )
        source_frames = source_frames
        source_timestamps = source_timestamps
        source_frame_indices = source_frame_indices
        
        # 提取掩码帧
        mask_frames = None
        mask_frame_indices = None
        if mask_vr is not None:
            if alignment_mode == "linspace_trim_padding":
                mask_frame_indices_for_eval = [
                    resize_frame_index(int(i), mask_total_frames, target_total_frames)
                    for i in eval_frame_indices
                ]
            else:
                mask_frame_indices_for_eval = [int(round(int(i) / target_fps * mask_fps)) for i in eval_frame_indices]
            if mask_frame_indices_for_eval and max(mask_frame_indices_for_eval) >= mask_total_frames:
                raise RuntimeError(
                    f"mask video is shorter than generated timeline: need mask frame "
                    f"{max(mask_frame_indices_for_eval)}, available={mask_total_frames}"
                )
            mask_frames, _, mask_frame_indices = extract_video_frames(
                mask_video_path, mask_frame_indices_for_eval, reader=mask_vr
            )
            mask_frames = mask_frames
            mask_frame_indices = mask_frame_indices
        
        expected_frames = len(eval_frame_indices)
        lengths = [len(target_frames), len(source_frames)] + ([len(mask_frames)] if mask_frames is not None else [])
        if any(length != expected_frames for length in lengths):
            raise RuntimeError(f"incomplete frame decode: expected={expected_frames}, decoded={lengths}")
        min_frames = expected_frames
        
        target_frames = target_frames[:min_frames]
        source_frames = source_frames[:min_frames]
        target_frame_indices = target_frame_indices[:min_frames]
        if mask_frames is not None:
            mask_frames = mask_frames[:min_frames]
        
        if min_frames == 0:
            return {
                'video_id': video_id,
                'ref_type': ref_type,
                'error': 'No valid frames',
                'face_sim_scores': [],
                'avg_face_sim': None,
                'face_sim_contrast_scores': [],
                'avg_face_sim_contrast': None,
                'lpips_scores': [],
                'avg_lpips': None,
                'ssim_scores': [],
                'avg_ssim': None,
                'pose_l2_distances': [],
                'avg_pose_l2_dist': None,
                'gaze_l2_distances': [],
                'avg_gaze_l2_dist': None,
                'gaze_cosine_similarities': [],
                'avg_gaze_cosine_sim': None,
                'exp_l2_distances': [],
                'avg_exp_l2_dist': None,
                'gamma_l2_distances': [],
                'avg_gamma_l2_dist': None,
                'id_retrieval_top1_accuracy': None,
                'id_retrieval_top5_accuracy': None,
                'id_retrieval_top10_accuracy': None,
                'id_retrieval_valid_frames_count': 0,
                'id_retrieval_total_frames_count': 0,
                'warping_error_short_list': [],
                'warping_error_long_list': [],
                'warping_error_mean_short': None,
                'warping_error_mean_long': None,
                'warping_error_mean_total': None,
                'frame_indices': []
            }
        
        source_shape = source_frames[0].shape[:2]
        # 对齐尺寸
        if target_frames[0].shape != source_frames[0].shape:
            height, width = target_frames[0].shape[:2]
            source_frames = [cv2.resize(frame, (width, height)) for frame in source_frames]
        
        if mask_frames is not None and mask_frames[0].shape != target_frames[0].shape:
            height, width = target_frames[0].shape[:2]
            mask_frames = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST) for frame in mask_frames]
        
        # 加载参考人脸图像
        ref_face = Image.open(ref_face_path).convert('RGB')
        
        # 初始化结果
        result = {
            'video_id': video_id,
            'base_video_id': base_video_id,  # 添加基础video_id（不含ref_type后缀）
            'ref_type': ref_type,  # ✅ 添加ref_type字段
            'num_frames': min_frames,
            'roi_source': 'face_boxes_json' if boxes is not None else 'mask_video',
            'source_frame_indices': source_frame_indices_for_eval,
            'metric_protocol_version': 'rgb_landmarks_gaze3d_directroi_v3',
            'face_sim_scores': [],
            'avg_face_sim': None,
            'face_sim_contrast_scores': [],
            'avg_face_sim_contrast': None,
            'lpips_scores': [],
            'avg_lpips': None,
            'ssim_scores': [],
            'avg_ssim': None,
            'pose_l2_distances': [],
            'avg_pose_l2_dist': None,
            'gaze_l2_distances': [],
            'avg_gaze_l2_dist': None,
            'gaze_cosine_similarities': [],
            'avg_gaze_cosine_sim': None,
            'exp_l2_distances': [],
            'avg_exp_l2_dist': None,
            'gamma_l2_distances': [],
            'avg_gamma_l2_dist': None,
            'id_retrieval_top1_accuracy': None,
            'id_retrieval_top5_accuracy': None,
            'id_retrieval_top10_accuracy': None,
            'id_retrieval_valid_frames_count': 0,
            'id_retrieval_total_frames_count': 0,
            'warping_error_short_list': [],
            'warping_error_long_list': [],
            'warping_error_mean_short': None,
            'warping_error_mean_long': None,
            'warping_error_mean_total': None,
            'frame_indices': target_frame_indices,
            'random_sampling': random_sampling,
            'frame_stride': frame_stride
        }
        
        # 处理人脸相关的评估
        if (
            self.config.enable_face_sim
            or self.config.enable_pose
            or self.config.enable_gaze
            or self.config.enable_exp_gamma
            or self.config.enable_id_retrieval
        ):
            # 过滤有效帧并裁剪人脸
            cropped_target_faces = []
            cropped_source_faces = []
            
            for i in range(min_frames):
                target_frame = target_frames[i]
                source_frame = source_frames[i]
                
                bbox = None
                if boxes is not None:
                    bbox = scaled_box(boxes[source_frame_indices_for_eval[i]], source_shape, target_frame.shape[:2])
                elif mask_frames is not None:
                    mask_frame = mask_frames[i]
                    bbox = calculate_bbox_from_mask(mask_frame)
                
                if bbox is None:
                    raise ValueError(f"missing/empty explicit ROI at generated frame {eval_frame_indices[i]}")
                if bbox is not None:
                    # 扩展边界框并裁剪人脸
                    expanded_bbox = expand_bbox(bbox, target_frame.shape[:2])
                    cropped_target_face = crop_face(target_frame, expanded_bbox)
                    cropped_source_face = crop_face(source_frame, expanded_bbox)
                    cropped_target_faces.append(cropped_target_face)
                    cropped_source_faces.append(cropped_source_face)
                else:
                    # 如果没有有效掩码，使用整个帧
                    cropped_target_faces.append(target_frame)
                    cropped_source_faces.append(source_frame)
            
            # 计算人脸相似度
            if self.config.enable_face_sim and len(cropped_target_faces) > 0:
                similarities = self.metrics_calculator.calculate_video_face_similarities(
                    cropped_target_faces, [ref_face, cropped_source_faces[0]], use_flip=True
                )
                result['avg_face_sim'], result['face_sim_scores'] = similarities[0]
                result['avg_face_sim_contrast'], result['face_sim_contrast_scores'] = similarities[1]
                result['face_sim_valid_frames'] = sum(v is not None for v in result['face_sim_scores'])

            # 计算姿态距离
            if self.config.enable_pose and len(cropped_target_faces) > 0:
                avg_pose_l2_dist, pose_l2_distances = self.metrics_calculator.calculate_video_pose_distance(
                    cropped_target_faces, cropped_source_faces
                )
                result['pose_l2_distances'] = pose_l2_distances
                result['avg_pose_l2_dist'] = avg_pose_l2_dist
            
            # 计算视线距离
            if self.config.enable_gaze and len(cropped_target_faces) > 0:
                avg_gaze_l2_dist, gaze_l2_distances, avg_gaze_cosine_sim, gaze_cosine_similarities = self.metrics_calculator.calculate_video_gaze_distance(
                    cropped_target_faces, cropped_source_faces
                )
                result['gaze_l2_distances'] = gaze_l2_distances
                result['avg_gaze_l2_dist'] = avg_gaze_l2_dist
                result['gaze_cosine_similarities'] = gaze_cosine_similarities
                result['avg_gaze_cosine_sim'] = avg_gaze_cosine_sim
            
            # 计算expression和gamma距离
            if self.config.enable_exp_gamma and len(cropped_target_faces) > 0:
                avg_exp_l2_dist, avg_gamma_l2_dist, exp_l2_distances, gamma_l2_distances = self.metrics_calculator.calculate_video_exp_gamma_distance(
                    cropped_target_faces, cropped_source_faces
                )
                result['exp_l2_distances'] = exp_l2_distances
                result['avg_exp_l2_dist'] = avg_exp_l2_dist
                result['gamma_l2_distances'] = gamma_l2_distances
                result['avg_gamma_l2_dist'] = avg_gamma_l2_dist
        
        # 计算LPIPS和SSIM指标
        if self.config.enable_lpips or self.config.enable_ssim:
            if os.getenv('RUN_MODE') == 'debug':
                breakpoint()
            back_mask_frames = None
            if mask_frames is not None:
                back_mask_frames = [255 - mask_frame for mask_frame in mask_frames]
            
            if self.config.enable_lpips:
                avg_lpips, lpips_scores = self.metrics_calculator.calculate_video_lpips(
                    target_frames, source_frames, back_mask_frames, back_mask_frames
                )
                result['lpips_scores'] = lpips_scores
                result['avg_lpips'] = avg_lpips
            
            if self.config.enable_ssim:
                avg_ssim, ssim_scores = self.metrics_calculator.calculate_video_ssim(
                    target_frames, source_frames, back_mask_frames, back_mask_frames
                )
                result['ssim_scores'] = ssim_scores
                result['avg_ssim'] = avg_ssim
        
        # 计算ID检索指标
        if self.config.enable_id_retrieval and ref_type in self.all_id_features_by_ref:
            # 使用对应ref_type的数据库
            all_id_features = self.all_id_features_by_ref[ref_type]
            ref_img_names = self.ref_img_names_by_ref[ref_type]
            
            # 获取目标参考图像名称
            target_ref_name = os.path.basename(ref_face_path) if ref_face_path else None
            
            try:
                id_retrieval_results = self.metrics_calculator.calculate_id_retrieval_metrics(
                    cropped_target_faces,
                    all_id_features,
                    ref_img_names,
                    target_ref_name
                )
                
                result['id_retrieval_top1_accuracy'] = id_retrieval_results['top1_accuracy']
                result['id_retrieval_top5_accuracy'] = id_retrieval_results['top5_accuracy']
                result['id_retrieval_top10_accuracy'] = id_retrieval_results['top10_accuracy']
                result['id_retrieval_valid_frames_count'] = id_retrieval_results['valid_frames_count']
                result['id_retrieval_total_frames_count'] = id_retrieval_results['total_frames_count']
                result['id_retrieval_target_ref_name'] = id_retrieval_results['target_ref_name']
                # result['id_retrieval_ref_type'] = ref_type  # 记录使用的ref类型
                
                # 可选：保存详细结果
                if self.config.save_detailed:
                    result['id_retrieval_frame_results'] = id_retrieval_results['frame_results']
                
                print(f"ID Retrieval Results ({ref_type}) - "
                      f"Top1: {id_retrieval_results['top1_accuracy']:.3f}, "
                      f"Top5: {id_retrieval_results['top5_accuracy']:.3f}, "
                      f"Top10: {id_retrieval_results['top10_accuracy']:.3f}")
                
            except Exception as e:
                print(f"Error calculating ID retrieval metrics: {e}")
                result['id_retrieval_error'] = str(e)
        
        # 计算Warping Error
        if self.config.enable_warping_error and mask_frames is not None:
            try:
                print(f"Calculating warping error for video: {video_id}")
                
                # 将numpy数组转换为PIL Images
                target_pil_frames = []
                source_pil_frames = []
                mask_pil_frames = []
                
                for i in range(min_frames):
                    # 转换target frames
                    target_img = Image.fromarray(target_frames[i].astype(np.uint8))
                    target_pil_frames.append(target_img)
                    
                    # 转换source frames
                    source_img = Image.fromarray(source_frames[i].astype(np.uint8))
                    source_pil_frames.append(source_img)
                    
                    # 转换mask frames (需要归一化到[0,1]范围)
                    mask_array = mask_frames[i].astype(np.float32) / 255.0
                    # 确保mask是[0,1]的二值化mask
                    mask_array = (mask_array > 0.5).astype(np.float32)
                    mask_img = Image.fromarray((mask_array * 255).astype(np.uint8))
                    mask_pil_frames.append(mask_img)
                
                # 计算warping error
                warping_results = self.metrics_calculator.calculate_warping_error(
                    orig_frames=source_pil_frames,
                    gen_frames=target_pil_frames,
                    face_masks=mask_pil_frames,
                    fb_thresh=self.config.warping_fb_thresh,
                    weights_path=self.config.warping_weights_path,
                    video_length=self.config.warping_video_length,
                    random_seed=self.config.warping_random_seed
                )
                
                result['warping_error_short_list'] = warping_results.get('short_list', [])
                result['warping_error_long_list'] = warping_results.get('long_list', [])
                result['warping_error_mean_short'] = warping_results.get('mean_short', None)
                result['warping_error_mean_long'] = warping_results.get('mean_long', None)
                result['warping_error_mean_total'] = warping_results.get('mean_total', None)
                
                print(f"Warping Error Results - Short: {warping_results.get('mean_short', 'N/A'):.4f}, "
                      f"Long: {warping_results.get('mean_long', 'N/A'):.4f}, "
                      f"Total: {warping_results.get('mean_total', 'N/A'):.4f}")
                
            except Exception as e:
                print(f"Error calculating warping error: {e}")
                result['warping_error_error'] = str(e)
        
        return result


def prepare_video_data(config: EvalConfig) -> dict:
    """准备视频数据，按ref类型分组"""
    print("正在查找视频对...")
    if config.mapping:
        rows = json.loads(Path(config.mapping).read_text())
        return {'sim': [{
            'video_id': f"{row['facebench_video_id']}_sim",
            'base_video_id': row['facebench_video_id'], 'ref_type': 'sim',
            'source_video_path': row['ref_video'], 'target_video_path': row['generated'],
            'mask_video_path': row['ref_video_facemask'], 'ref_face_path': row['ref_image'],
            'max_frames': config.max_frames, 'random_sampling': config.random_sampling,
            'frame_stride': config.frame_stride,
        } for row in rows], 'mid': [], 'diff': []}

    source_pairs = find_video_pairs_new(
        config.source_video_dir,
        config.use_align,
        config.video_num
    )
    
    print(f"找到 {len(source_pairs)} 个源视频")
    
    # 获取对应的目标视频
    target_videos_list = get_corresponding_target_videos_new(source_pairs, config.target_video_dir)
    
    # 按ref类型分组准备评估数据
    video_data_by_ref = {
        'sim': [],
        'mid': [],
        'diff': []
    }
    
    for source_pair, target_videos in zip(source_pairs, target_videos_list):
        if target_videos is None:
            continue
        
        source_video_path = source_pair[0]
        mask_video_path = source_pair[1]
        ref_paths = source_pair[2]
        video_id = source_pair[3]
        
        # 为每个ref类型创建评估数据
        for ref_type in ref_paths.keys():
            if ref_type not in target_videos:
                continue
            
            video_data = {
                'video_id': f"{video_id}_{ref_type}",
                'base_video_id': video_id,
                'ref_type': ref_type,
                'source_video_path': source_video_path,
                'target_video_path': target_videos[ref_type],
                'mask_video_path': mask_video_path,
                'ref_face_path': ref_paths[ref_type],
                'max_frames': config.max_frames,
                'random_sampling': config.random_sampling,
                'frame_stride': config.frame_stride
            }
            video_data_by_ref[ref_type].append(video_data)
    
    video_data_by_ref = {
        'sim': video_data_by_ref['sim'],
        'mid': [],
        'diff': video_data_by_ref['diff']
    }  # NOTE only for diff
    
    # 打印统计信息
    for ref_type, video_list in video_data_by_ref.items():
        print(f"准备评估 {ref_type}: {len(video_list)} 个视频")
    
    return video_data_by_ref


def collect_all_results(config: EvalConfig) -> List[dict]:
    """从单个结果文件中收集所有评估结果"""
    single_result_dir = os.path.join(config.output_dir, config.single_result_dir)
    
    if not os.path.exists(single_result_dir):
        print(f"Single result directory not found: {single_result_dir}")
        return []
    
    results = []
    result_files = [f for f in os.listdir(single_result_dir) if f.endswith('.json')]
    
    print(f"Found {len(result_files)} result files in {single_result_dir}")
    
    for result_file in result_files:
        file_path = os.path.join(single_result_dir, result_file)
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Error loading result file {result_file}: {e}")
    
    return results


def save_results_by_ref(results_by_ref: dict, config: EvalConfig):
    """按ref类型保存评估结果，并生成汇总结果"""
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 为每个ref类型生成单独的汇总
    all_summaries = {}
    
    for ref_type, results in results_by_ref.items():
        if not results:
            continue
        
        print(f"\n=== 处理 {ref_type} 类型的结果 ===")
        
        # 汇总当前ref类型的结果
        all_face_sims = []
        all_face_sims_src = []
        all_lpips = []
        all_ssims = []
        all_pose_dists = []
        all_gaze_l2_dists = []
        all_gaze_cosine_sims = []
        all_exp_l2_dists = []
        all_gamma_l2_dists = []
        all_id_retrieval_top1 = []
        all_id_retrieval_top5 = []
        all_id_retrieval_top10 = []
        all_warping_error_short = []
        all_warping_error_long = []
        all_warping_error_total = []
        video_results = {}
        
        for result in results:
            video_id = result['video_id']
            video_results[video_id] = result
            
            # 收集各项指标
            if result.get('avg_face_sim') is not None:
                all_face_sims.append(result['avg_face_sim'])
            if result.get('avg_face_sim_contrast') is not None:
                all_face_sims_src.append(result['avg_face_sim_contrast'])
            if result.get('avg_lpips') is not None:
                all_lpips.append(result['avg_lpips'])
            if result.get('avg_ssim') is not None:
                all_ssims.append(result['avg_ssim'])
            if result.get('avg_pose_l2_dist') is not None:
                all_pose_dists.append(result['avg_pose_l2_dist'])
            if result.get('avg_gaze_l2_dist') is not None:
                all_gaze_l2_dists.append(result['avg_gaze_l2_dist'])
            if result.get('avg_gaze_cosine_sim') is not None:
                all_gaze_cosine_sims.append(result['avg_gaze_cosine_sim'])
            if result.get('avg_exp_l2_dist') is not None:
                all_exp_l2_dists.append(result['avg_exp_l2_dist'])
            if result.get('avg_gamma_l2_dist') is not None:
                all_gamma_l2_dists.append(result['avg_gamma_l2_dist'])
            if result.get('id_retrieval_top1_accuracy') is not None:
                all_id_retrieval_top1.append(result['id_retrieval_top1_accuracy'])
            if result.get('id_retrieval_top5_accuracy') is not None:
                all_id_retrieval_top5.append(result['id_retrieval_top5_accuracy'])
            if result.get('id_retrieval_top10_accuracy') is not None:
                all_id_retrieval_top10.append(result['id_retrieval_top10_accuracy'])
            if result.get('warping_error_mean_short') is not None and not np.isnan(result['warping_error_mean_short']):
                all_warping_error_short.append(result['warping_error_mean_short'])
            if result.get('warping_error_mean_long') is not None and not np.isnan(result['warping_error_mean_long']):
                all_warping_error_long.append(result['warping_error_mean_long'])
            if result.get('warping_error_mean_total') is not None and not np.isnan(result['warping_error_mean_total']):
                all_warping_error_total.append(result['warping_error_mean_total'])
        
        # 计算当前ref类型的统计
        summary = {
            'metric_protocol_version': 'rgb_landmarks_gaze3d_directroi_v3',
            'gaze_cosine_definition': '3D unit direction cosine from degree angles',
            'deep3d_landmark_model': os.environ.get('FACEBENCH_LANDMARK_MODEL'),
            'ref_type': ref_type,
            'total_videos': len(results),
            'case_count': len(results),
            'failure_count': sum(bool(row.get('error')) for row in results),
            'config': {
                'enable_face_sim': config.enable_face_sim,
                'enable_lpips': config.enable_lpips,
                'enable_ssim': config.enable_ssim,
                'enable_pose': config.enable_pose,
                'enable_gaze': config.enable_gaze,
                'enable_exp_gamma': config.enable_exp_gamma,
                'enable_id_retrieval': config.enable_id_retrieval,
                'enable_warping_error': config.enable_warping_error,
                'random_sampling': config.random_sampling,
                'frame_stride': config.frame_stride
            },
            'metrics': {}
        }
        
        # 添加各项指标的统计
        if config.enable_face_sim and all_face_sims:
            summary['metrics']['face_similarity'] = {
                'valid_videos': len(all_face_sims),
                'mean': float(np.mean(all_face_sims)),
                'std': float(np.std(all_face_sims)),
                'min': float(np.min(all_face_sims)),
                'max': float(np.max(all_face_sims))
            }
        
        if config.enable_face_sim and all_face_sims_src:
            summary['metrics']['face_similarity_src'] = {
                'valid_videos': len(all_face_sims_src),
                'mean': float(np.mean(all_face_sims_src)),
                'std': float(np.std(all_face_sims_src)),
                'min': float(np.min(all_face_sims_src)),
                'max': float(np.max(all_face_sims_src))
            }
        
        if config.enable_lpips and all_lpips:
            summary['metrics']['lpips'] = {
                'valid_videos': len(all_lpips),
                'mean': float(np.mean(all_lpips)),
                'std': float(np.std(all_lpips)),
                'min': float(np.min(all_lpips)),
                'max': float(np.max(all_lpips))
            }
        
        if config.enable_ssim and all_ssims:
            summary['metrics']['ssim'] = {
                'valid_videos': len(all_ssims),
                'mean': float(np.mean(all_ssims)),
                'std': float(np.std(all_ssims)),
                'min': float(np.min(all_ssims)),
                'max': float(np.max(all_ssims))
            }
        
        if config.enable_pose and all_pose_dists:
            summary['metrics']['pose_distance'] = {
                'valid_videos': len(all_pose_dists),
                'mean': float(np.mean(all_pose_dists)),
                'std': float(np.std(all_pose_dists)),
                'min': float(np.min(all_pose_dists)),
                'max': float(np.max(all_pose_dists))
            }
        
        if config.enable_gaze and all_gaze_l2_dists:
            summary['metrics']['gaze_l2_distance'] = {
                'valid_videos': len(all_gaze_l2_dists),
                'mean': float(np.mean(all_gaze_l2_dists)),
                'std': float(np.std(all_gaze_l2_dists)),
                'min': float(np.min(all_gaze_l2_dists)),
                'max': float(np.max(all_gaze_l2_dists))
            }
        
        if config.enable_gaze and all_gaze_cosine_sims:
            summary['metrics']['gaze_cosine_similarity'] = {
                'valid_videos': len(all_gaze_cosine_sims),
                'mean': float(np.mean(all_gaze_cosine_sims)),
                'std': float(np.std(all_gaze_cosine_sims)),
                'min': float(np.min(all_gaze_cosine_sims)),
                'max': float(np.max(all_gaze_cosine_sims))
            }
        
        if config.enable_exp_gamma and all_exp_l2_dists:
            summary['metrics']['exp_l2_distance'] = {
                'valid_videos': len(all_exp_l2_dists),
                'mean': float(np.mean(all_exp_l2_dists)),
                'std': float(np.std(all_exp_l2_dists)),
                'min': float(np.min(all_exp_l2_dists)),
                'max': float(np.max(all_exp_l2_dists))
            }
        
        if config.enable_exp_gamma and all_gamma_l2_dists:
            summary['metrics']['gamma_l2_distance'] = {
                'valid_videos': len(all_gamma_l2_dists),
                'mean': float(np.mean(all_gamma_l2_dists)),
                'std': float(np.std(all_gamma_l2_dists)),
                'min': float(np.min(all_gamma_l2_dists)),
                'max': float(np.max(all_gamma_l2_dists))
            }
        
        if config.enable_id_retrieval and all_id_retrieval_top1:
            summary['metrics']['id_retrieval_top1_accuracy'] = {
                'valid_videos': len(all_id_retrieval_top1),
                'mean': float(np.mean(all_id_retrieval_top1)),
                'std': float(np.std(all_id_retrieval_top1)),
                'min': float(np.min(all_id_retrieval_top1)),
                'max': float(np.max(all_id_retrieval_top1))
            }
        
        if config.enable_id_retrieval and all_id_retrieval_top5:
            summary['metrics']['id_retrieval_top5_accuracy'] = {
                'valid_videos': len(all_id_retrieval_top5),
                'mean': float(np.mean(all_id_retrieval_top5)),
                'std': float(np.std(all_id_retrieval_top5)),
                'min': float(np.min(all_id_retrieval_top5)),
                'max': float(np.max(all_id_retrieval_top5))
            }
        
        if config.enable_id_retrieval and all_id_retrieval_top10:
            summary['metrics']['id_retrieval_top10_accuracy'] = {
                'valid_videos': len(all_id_retrieval_top10),
                'mean': float(np.mean(all_id_retrieval_top10)),
                'std': float(np.std(all_id_retrieval_top10)),
                'min': float(np.min(all_id_retrieval_top10)),
                'max': float(np.max(all_id_retrieval_top10))
            }
        
        if config.enable_warping_error and all_warping_error_short:
            summary['metrics']['warping_error_short'] = {
                'valid_videos': len(all_warping_error_short),
                'mean': float(np.mean(all_warping_error_short)),
                'std': float(np.std(all_warping_error_short)),
                'min': float(np.min(all_warping_error_short)),
                'max': float(np.max(all_warping_error_short))
            }
        
        if config.enable_warping_error and all_warping_error_long:
            summary['metrics']['warping_error_long'] = {
                'valid_videos': len(all_warping_error_long),
                'mean': float(np.mean(all_warping_error_long)),
                'std': float(np.std(all_warping_error_long)),
                'min': float(np.min(all_warping_error_long)),
                'max': float(np.max(all_warping_error_long))
            }
        
        if config.enable_warping_error and all_warping_error_total:
            summary['metrics']['warping_error_total'] = {
                'valid_videos': len(all_warping_error_total),
                'mean': float(np.mean(all_warping_error_total)),
                'std': float(np.std(all_warping_error_total)),
                'min': float(np.min(all_warping_error_total)),
                'max': float(np.max(all_warping_error_total))
            }
        
        required_metrics = []
        for enabled, names in [
            (config.enable_face_sim, ['face_similarity']),
            (config.enable_pose, ['pose_distance']),
            (config.enable_gaze, ['gaze_l2_distance', 'gaze_cosine_similarity']),
            (config.enable_exp_gamma, ['exp_l2_distance', 'gamma_l2_distance']),
        ]:
            if enabled:
                required_metrics.extend(names)
        summary['missing_metrics'] = [name for name in required_metrics if name not in summary['metrics']]
        all_summaries[ref_type] = summary
        
        # 保存当前ref类型的汇总结果
        summary_path = os.path.join(config.output_dir, f"evaluation_summary_{ref_type}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"{ref_type} 类型汇总结果已保存到: {summary_path}")
        
        # 打印结果
        print(f"\n{ref_type} 类型结果:")
        for metric_name, metric_data in summary['metrics'].items():
            print(f"  平均{metric_name}: {metric_data['mean']:.4f} (std: {metric_data['std']:.4f})")
        
        # 保存详细结果
        if config.save_detailed:
            detailed_path = os.path.join(config.output_dir, f"evaluation_detailed_{ref_type}.json")
            
            # 转换numpy类型为普通类型以便JSON序列化
            for video_id, result in video_results.items():
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        result[key] = value.tolist()
                    elif isinstance(value, np.floating):
                        result[key] = float(value)
                    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.floating):
                        result[key] = [float(v) for v in value]
            
            with open(detailed_path, 'w') as f:
                json.dump(video_results, f, indent=2)
            
            print(f"{ref_type} 类型详细结果已保存到: {detailed_path}")
    
    # 生成总体汇总（合并所有ref类型）
    overall_summary = {
        'total_ref_types': len(all_summaries),
        'ref_types': list(all_summaries.keys()),
        'by_ref_type': all_summaries,
        'overall_metrics': {}
    }
    
    # 计算所有ref类型的平均指标
    for metric_name in ['face_similarity', 'face_similarity_src', 'lpips', 'ssim', 
                       'pose_distance', 'gaze_l2_distance', 'gaze_cosine_similarity',
                       'exp_l2_distance', 'gamma_l2_distance',
                       'id_retrieval_top1_accuracy', 'id_retrieval_top5_accuracy', 
                       'id_retrieval_top10_accuracy',
                       'warping_error_short', 'warping_error_long', 'warping_error_total']:
        metric_values = []
        for ref_type, summary in all_summaries.items():
            if metric_name in summary['metrics']:
                metric_values.append(summary['metrics'][metric_name]['mean'])
        
        if metric_values:
            overall_summary['overall_metrics'][metric_name] = {
                'mean_across_ref_types': float(np.mean(metric_values)),
                'std_across_ref_types': float(np.std(metric_values)),
                'min_across_ref_types': float(np.min(metric_values)),
                'max_across_ref_types': float(np.max(metric_values))
            }
    
    # 保存总体汇总
    overall_path = os.path.join(config.output_dir, "evaluation_summary_overall.json")
    with open(overall_path, 'w') as f:
        json.dump(overall_summary, f, indent=2, default=str)
    
    print(f"\n总体汇总结果已保存到: {overall_path}")
    print("\n=== 总体结果 (跨ref类型平均) ===")
    for metric_name, metric_data in overall_summary['overall_metrics'].items():
        print(f"  {metric_name}: {metric_data['mean_across_ref_types']:.4f} "
              f"(std: {metric_data['std_across_ref_types']:.4f})")


def collect_all_results_by_ref(config: EvalConfig) -> dict:
    """从单个结果文件中收集所有评估结果，按ref类型分组"""
    single_result_dir = os.path.join(config.output_dir, config.single_result_dir)
    
    if not os.path.exists(single_result_dir):
        print(f"Single result directory not found: {single_result_dir}")
        return {'sim': [], 'mid': [], 'diff': []}
    
    results_by_ref = {'sim': [], 'mid': [], 'diff': []}
    result_files = [f for f in os.listdir(single_result_dir) if f.endswith('.json')]
    
    print(f"Found {len(result_files)} result files in {single_result_dir}")
    
    for result_file in result_files:
        file_path = os.path.join(single_result_dir, result_file)
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                ref_type = result.get('ref_type')
                if ref_type in results_by_ref:
                    results_by_ref[ref_type].append(result)
        except Exception as e:
            print(f"Error loading result file {result_file}: {e}")
    
    for ref_type, results in results_by_ref.items():
        print(f"Collected {len(results)} results for {ref_type}")
    
    return results_by_ref


def main(config: EvalConfig):
    """主函数"""
    print("开始综合视频质量评估...")
    print(f"配置: {config}")
    
    # ...existing code for path checking...
    
    ENV_LIMITS = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "OPENCV_OPENMP_RUNTIME": "disabled",
        "OMP_WAIT_POLICY": "PASSIVE",
        "KMP_BLOCKTIME": "0",
        "TORCH_CUDA_ARCH_LIST": "9.0",
    }

    if not ray.is_initialized():
        ray_temp_dir = os.environ.get(
            "RAY_TEMP_DIR",
            f"/tmp/swapface_ray_{os.getpid()}",
        )
        os.makedirs(ray_temp_dir, exist_ok=True)
        ray.init(
            include_dashboard=False,
            num_cpus=16,
            num_gpus=config.num_gpus,
            runtime_env={"env_vars": ENV_LIMITS},
            _temp_dir=ray_temp_dir,
            _node_ip_address="127.0.0.1",
        )
    
    try:
        # 准备视频数据（按ref类型分组）
        video_data_by_ref = prepare_video_data(config)
        
        total_videos = sum(len(v) for v in video_data_by_ref.values())
        if total_videos == 0:
            print("没有找到有效的视频对，退出。")
            raise ValueError("no evaluation video pairs")
        
        # 如果启用resume，检查已有结果
        if config.enable_resume:
            single_result_dir = os.path.join(config.output_dir, config.single_result_dir)
            os.makedirs(single_result_dir, exist_ok=True)
            
            existing_results_by_ref = collect_all_results_by_ref(config)
            
            for ref_type in video_data_by_ref.keys():
                existing_count = len(existing_results_by_ref.get(ref_type, []))
                total_count = len(video_data_by_ref[ref_type])
                print(f"Resume模式 ({ref_type}): 总计 {total_count} 个视频，已完成 {existing_count} 个")
        
        # 创建远程评估器
        evaluators = []
        for i in range(config.num_gpus):
            evaluator = ComprehensiveEvaluator.remote(config)
            evaluators.append(evaluator)
        
        # 合并所有ref类型的视频数据
        all_video_data = []
        for ref_type, video_list in video_data_by_ref.items():
            all_video_data.extend(video_list)
        
        # 分批处理视频
        batch_size = config.batch_size
        video_batches = [
            all_video_data[i:i + batch_size] 
            for i in range(0, len(all_video_data), batch_size)
        ]
        
        print(f"\n将 {len(all_video_data)} 个视频分为 {len(video_batches)} 批，每批 {batch_size} 个")
        
        # 分配任务到不同GPU
        futures = []
        for i, batch in enumerate(video_batches):
            evaluator = evaluators[i % config.num_gpus]
            future = evaluator.evaluate_video_batch.remote(batch)
            futures.append(future)
        
        # 收集结果
        all_results = []
        for future in tqdm(futures, desc="等待评估完成"):
            batch_results = ray.get(future)
            all_results.extend(batch_results)
        
        # 从单个结果文件中收集所有结果并按ref类型分组
        print("\n正在收集所有结果文件...")
        all_collected_results_by_ref = {key: [] for key in video_data_by_ref}
        for result in all_results:
            all_collected_results_by_ref[result.get('ref_type', 'sim')].append(result)
        
        # 保存结果（按ref类型分别保存，并生成总体汇总）
        save_results_by_ref(all_collected_results_by_ref, config)
        
        if any(result.get("error") for result in all_results):
            raise RuntimeError("FaceBench cases failed; see per-case errors and failure_count")
        for ref_type, rows in all_collected_results_by_ref.items():
            if rows:
                saved = json.loads(Path(config.output_dir, f'evaluation_summary_{ref_type}.json').read_text())
                if saved.get('missing_metrics'):
                    raise RuntimeError(f"selected metrics have no valid results: {saved['missing_metrics']}")
        print("\n评估完成！")
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    config = tyro.cli(EvalConfig)
    main(config)
