import os
import re
import hashlib
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List
from kornia.augmentation import CenterCrop
# 假设 transform 能够处理多通道输入 (C, H, W)，如果你的 GenericRandomTransform 
# 内部写死了只处理单通道深度图，可能需要微调，但通常 Kornia/TorchVision 都能处理多通道。
from .generic_transform import GenericRandomTransform
from util.helper import metric_to_ips

import OpenEXR
import Imath

def read_exr(file_path):
    """
    使用OpenEXR库读取.exr文件并返回一个NumPy数组。
    """
    if not OpenEXR.isOpenExrFile(file_path):
        raise IOError(f"文件不是一个有效的EXR文件: {file_path}")

    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()

    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels_info = header['channels']
    channel_names = sorted(channels_info.keys())
    
    if not channel_names:
        raise ValueError(f"EXR文件没有任何通道: {file_path}")

    first_channel_type = channels_info[channel_names[0]]
    if first_channel_type.type == Imath.PixelType(Imath.PixelType.FLOAT):
        dtype = np.float32
    elif first_channel_type.type == Imath.PixelType(Imath.PixelType.HALF):
        dtype = np.float16
    else:
        raise TypeError(f"不支持的EXR数据类型: {first_channel_type.type}")

    all_channels_bytes = exr_file.channels(channel_names)

    np_channels = []
    for i, name in enumerate(channel_names):
        channel_data = np.frombuffer(all_channels_bytes[i], dtype=dtype)
        channel_data = channel_data.reshape(height, width)
        np_channels.append(channel_data)

    image_np = np.stack(np_channels, axis=-1)
    
    # 稍微放宽全0检查，避免浮点数精度极低的情况误判，但通常全0就是0
    if np.allclose(image_np, 0):
        raise ValueError(f"❌ EXR文件 {file_path} 读取后全为零！")

    return image_np


class HyperspectralDepthDataset(Dataset):
    def __init__(self, base_dir: str, scene_folders: List[str], image_size: Tuple[int, int], hs_channels: int,
                 is_training: bool = True, randcrop: bool = False, augment: bool = False,
                 min_depth: float = 0.4, max_depth: float = 2.0,
                 use_exr_cache: bool = False, exr_cache_dir: str = '',
                 patch_filter: bool = True, min_valid_ratio: float = 0.12,
                 min_depth_range_ips: float = 0.10, max_crop_retries: int = 8,
                 patch_filter_stride: int = 4):
        
        super().__init__()
        self.is_training = is_training

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.transform = GenericRandomTransform(image_size, randcrop, augment, hs_channels)
        self.centercrop = CenterCrop(image_size)
        self.hs_channels = int(hs_channels)

        self.min_depth = min_depth
        self.max_depth = max_depth

        self.image_size = tuple(image_size)
        self.randcrop = bool(randcrop)
        self.augment = bool(augment)

        self.patch_filter = bool(patch_filter and self.is_training and self.randcrop)
        self.min_valid_ratio = float(min_valid_ratio)
        self.min_depth_range_ips = float(min_depth_range_ips)
        self.max_crop_retries = max(1, int(max_crop_retries))
        self.patch_filter_stride = max(1, int(patch_filter_stride))

        self.use_exr_cache = use_exr_cache
        if not exr_cache_dir:
            exr_cache_dir = os.path.join(base_dir, '.exr_cache_npy_v1')
        self.exr_cache_dir = exr_cache_dir
        if self.use_exr_cache:
            os.makedirs(self.exr_cache_dir, exist_ok=True)
            mode = 'train' if self.is_training else 'val'
            print(f"[Dataset Cache] mode={mode}, dir={self.exr_cache_dir}")

        if self.patch_filter:
            print(
                f"[Patch Filter] enabled (fast), min_valid_ratio={self.min_valid_ratio:.3f}, "
                f"min_depth_range_ips={self.min_depth_range_ips:.3f}, retries={self.max_crop_retries}, "
                f"stride={self.patch_filter_stride}"
            )

        self.sample_pairs = []
        for folder_name in scene_folders:
            match = re.search(r'\d+', folder_name)
            if not match: continue
            scene_num = match.group(0).zfill(2)

            hs_path = os.path.join(base_dir, folder_name, f'scene{scene_num}_hs.exr')
            depth_path = os.path.join(base_dir, folder_name, f'scene{scene_num}_depth_map.exr')

            if os.path.exists(hs_path) and os.path.exists(depth_path):
                self.sample_pairs.append({'hs_path': hs_path, 'depth_path': depth_path, 'id': f'scene_{scene_num}'})
            else:
                # 仅在调试时打印，避免刷屏
                pass

    def __len__(self):
        return len(self.sample_pairs)

    def _cache_file_path(self, exr_path: str) -> str:
        stat = os.stat(exr_path)
        key_src = f"{exr_path}|{stat.st_mtime_ns}|{stat.st_size}"
        key = hashlib.sha1(key_src.encode('utf-8')).hexdigest()
        return os.path.join(self.exr_cache_dir, f"{key}.npy")

    def _read_exr_with_cache(self, exr_path: str) -> np.ndarray:
        if not self.use_exr_cache:
            return read_exr(exr_path)

        cache_path = self._cache_file_path(exr_path)
        if os.path.exists(cache_path):
            return np.load(cache_path, allow_pickle=False)

        image_np = read_exr(exr_path)
        tmp_path = f"{cache_path}.tmp.{os.getpid()}.npy"
        np.save(tmp_path, image_np)
        try:
            os.replace(tmp_path, cache_path)
        except OSError:
            # 多进程下可能并发写入同一缓存文件，保留先写入者即可。
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return image_np

    def _patch_quality_stats(self, depth_patch: torch.Tensor, mask_patch: torch.Tensor) -> Tuple[float, float]:
        valid = mask_patch > 0.5
        valid_ratio = float(valid.float().mean().item())
        if valid.sum() == 0:
            return valid_ratio, 0.0
        valid_depth = depth_patch[valid]
        depth_range = float((valid_depth.max() - valid_depth.min()).item())
        return valid_ratio, depth_range

    def _sample_random_crop_window(self, full_h: int, full_w: int) -> Tuple[int, int]:
        crop_h, crop_w = self.image_size
        if full_h <= crop_h:
            top = 0
        else:
            top = int(torch.randint(0, full_h - crop_h + 1, (1,)).item())

        if full_w <= crop_w:
            left = 0
        else:
            left = int(torch.randint(0, full_w - crop_w + 1, (1,)).item())
        return top, left

    def _crop_window(self, x: torch.Tensor, top: int, left: int) -> torch.Tensor:
        crop_h, crop_w = self.image_size
        return x[..., top:top + crop_h, left:left + crop_w]

    def _apply_random_flips(self, x: torch.Tensor, do_vflip: bool, do_hflip: bool) -> torch.Tensor:
        if do_vflip:
            x = torch.flip(x, dims=[-2])
        if do_hflip:
            x = torch.flip(x, dims=[-1])
        return x

    def __getitem__(self, idx):
        sample = self.sample_pairs[idx]
        hs_path = sample['hs_path']
        depth_path = sample['depth_path']
        sample_id = sample['id']
        
        try:
            hs_image = self._read_exr_with_cache(hs_path)
            depth_map = self._read_exr_with_cache(depth_path)
        except Exception as e:
            raise IOError(f"无法读取文件: {sample_id} \n错误: {e}")

        hs_image = hs_image.astype(np.float32)
        depth_map = depth_map.astype(np.float32)

        if hs_image.ndim != 3:
            raise ValueError(f"高光谱图像维度异常: {sample_id}, shape={hs_image.shape}")
        if hs_image.shape[2] < self.hs_channels:
            raise ValueError(
                f"高光谱通道数不足: {sample_id}, got={hs_image.shape[2]}, required={self.hs_channels}"
            )
        # 数据原始波段按 420nm -> 700nm 递增排列，这里只保留 420nm -> 660nm 的前 25 个波段。
        hs_image = hs_image[:, :, :self.hs_channels]

        # 确保深度图是 (H, W)
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(-1)
        
        # --- 深度单位转换 (mm -> m) ---
        depth_map = depth_map / 1000.0

        # ============================================================
        # 步骤 A: 生成 Mask (在归一化造成负数之前)
        # ============================================================
        # 【IPS 迁移】背景是0，物体是0.4~2.0。
        # 任何低于 min_depth 的都是背景，标记为 0；有效物体标记为 1。
        # 这个 Mask 稍后会在 Loss 计算时用到，确保网络不学背景的虚假值。
        valid_mask = (depth_map > self.min_depth - 1e-3).astype(np.float32)

        # ============================================================
        # 步骤 B: 高光谱图像处理 (保持你原有的去异常值逻辑)
        # ============================================================
        SANITY_THRESHOLD = 10000.0 
        min_hs = 0.0 # 假设最小光强是0
        
        # 你的逻辑：如果最大值异常大，寻找正常的第二大值
        if np.max(hs_image) > SANITY_THRESHOLD:
            valid_pixels = hs_image < SANITY_THRESHOLD
            if np.any(valid_pixels):
                real_max = np.max(hs_image[valid_pixels])
            else:
                real_max = 1.0
            # 截断异常值
            hs_image = np.clip(hs_image, min_hs, real_max)
            max_hs = real_max
        else:
            max_hs = np.max(hs_image)

        # HS 归一化
        if max_hs > min_hs:
            hs_image = (hs_image - min_hs) / (max_hs - min_hs)
        
        # ============================================================
        # 步骤 C: 深度图归一化 (【IPS 体系】使用逆深度均匀化)
        # ============================================================
        # 【迁移到 IPS】线性深度分布在远场会导致 PSF 差异很小。
        # 改用逆深度 (Inverse Perspective Sampling) 使深度分辨率均匀。
        #
        # 数学：
        #   物理深度 d ∈ [0.4m, 2.0m]
        #   逆深度 d_inv = 1/d ∈ [0.5, 2.5]
        #   归一化: d_norm = (1/d - 1/d_max) / (1/d_min - 1/d_max)
        #        = (max_depth * d - max_depth * min_depth) / ((max_depth - min_depth) * d)
        #
        # 边界条件：
        #   d=min_depth(0.4m): d_norm → 0.0（最近处）
        #   d=max_depth(2.0m): d_norm → 1.0（最远处）
        #   背景 d=0: 设为 1.0（或任意值，Mask 会排除）
        #
        # 使用 util.helper.metric_to_ips() 函数进行转换
        # ============================================================
        
        # 将 depth_map 转为 torch.Tensor 以使用 metric_to_ips
        depth_tensor = torch.from_numpy(depth_map).float()

        # 创建背景 mask（depth < min_depth 的像素）
        valid_mask_bool = depth_tensor >= self.min_depth - 1e-3

        # depth_metric: 物理米制深度，背景 clamp 到 min_depth（配合 mask 抑制无效区域）
        depth_metric_tensor = torch.clamp(depth_tensor, self.min_depth, self.max_depth)

        # 对有效前景像素使用 metric_to_ips 函数进行 IPS 归一化
        depth_safe = torch.where(valid_mask_bool, depth_tensor, torch.tensor(self.min_depth))
        ips_depth_tensor = metric_to_ips(depth_safe, self.min_depth, self.max_depth)

        # 确保在 [0, 1] 范围内，并转回 numpy
        depth_map = torch.clamp(ips_depth_tensor, 0.0, 1.0).numpy()
        depth_metric_np = depth_metric_tensor.numpy()

        # ============================================================
        # 步骤 D: 转 Tensor 并处理 Transform
        # ============================================================
        # HS: [H, W, C] -> [C, H, W]
        hs_tensor = torch.from_numpy(hs_image).permute(2, 0, 1).float()
        
        # Depth & Mask: [H, W] -> [1, H, W]
        depth_tensor = torch.from_numpy(depth_map).unsqueeze(0).float()
        depth_metric_tensor = torch.from_numpy(depth_metric_np).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(valid_mask).unsqueeze(0).float()

        # 增加 batch 维度以适配 transform: [1, C, H, W]
        hs_tensor = hs_tensor.unsqueeze(0)
        depth_tensor = depth_tensor.unsqueeze(0)
        depth_metric_tensor = depth_metric_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)

        if self.is_training:
            hs_base = hs_tensor
            depth_base = depth_tensor
            depth_metric_base = depth_metric_tensor
            mask_base = mask_tensor

            # 拼接 depth_ips / depth_metric / mask，确保几何变换一致
            depth_mask_cat_base = torch.cat([depth_base, depth_metric_base, mask_base], dim=1)

            if self.patch_filter:
                _, _, full_h, full_w = depth_base.shape
                crop_h, crop_w = self.image_size
                stride = self.patch_filter_stride

                if stride > 1:
                    depth_proxy = depth_base[:, :, ::stride, ::stride]
                    mask_proxy = mask_base[:, :, ::stride, ::stride]
                    crop_h_proxy = max(1, math.ceil(crop_h / stride))
                    crop_w_proxy = max(1, math.ceil(crop_w / stride))
                else:
                    depth_proxy = None
                    mask_proxy = None
                    crop_h_proxy = 0
                    crop_w_proxy = 0

                top, left = self._sample_random_crop_window(full_h, full_w)

                for _ in range(self.max_crop_retries):
                    if depth_proxy is not None and mask_proxy is not None:
                        top_proxy = top // stride
                        left_proxy = left // stride
                        depth_try_proxy = depth_proxy[
                            ..., top_proxy:top_proxy + crop_h_proxy, left_proxy:left_proxy + crop_w_proxy
                        ]
                        mask_try_proxy = mask_proxy[
                            ..., top_proxy:top_proxy + crop_h_proxy, left_proxy:left_proxy + crop_w_proxy
                        ]
                        valid_ratio_proxy, depth_range_proxy = self._patch_quality_stats(
                            depth_try_proxy, mask_try_proxy
                        )
                        if (
                            valid_ratio_proxy < self.min_valid_ratio
                            or depth_range_proxy < self.min_depth_range_ips
                        ):
                            top, left = self._sample_random_crop_window(full_h, full_w)
                            continue

                    depth_try = self._crop_window(depth_base, top, left)
                    mask_try = self._crop_window(mask_base, top, left)
                    valid_ratio, depth_range = self._patch_quality_stats(depth_try, mask_try)

                    if valid_ratio >= self.min_valid_ratio and depth_range >= self.min_depth_range_ips:
                        break
                    top, left = self._sample_random_crop_window(full_h, full_w)

                hs_tensor = self._crop_window(hs_base, top, left)
                depth_tensor = self._crop_window(depth_base, top, left)
                depth_metric_tensor = self._crop_window(depth_metric_base, top, left)
                mask_tensor = (self._crop_window(mask_base, top, left) > 0.5).float()

                if self.augment:
                    do_vflip = bool(torch.rand(1).item() < 0.5)
                    do_hflip = bool(torch.rand(1).item() < 0.5)
                    hs_tensor = self._apply_random_flips(hs_tensor, do_vflip, do_hflip)
                    depth_tensor = self._apply_random_flips(depth_tensor, do_vflip, do_hflip)
                    depth_metric_tensor = self._apply_random_flips(depth_metric_tensor, do_vflip, do_hflip)
                    mask_tensor = self._apply_random_flips(mask_tensor, do_vflip, do_hflip)
            else:
                hs_tensor, depth_mask_cat = self.transform(hs_base, depth_mask_cat_base)
                depth_tensor = depth_mask_cat[:, 0:1, :, :]
                depth_metric_tensor = depth_mask_cat[:, 1:2, :, :]
                mask_tensor = (depth_mask_cat[:, 2:3, :, :] > 0.5).float()

        else:
            hs_tensor = self.centercrop(hs_tensor)
            depth_tensor = self.centercrop(depth_tensor)
            depth_metric_tensor = self.centercrop(depth_metric_tensor)
            mask_tensor = self.centercrop(mask_tensor)

        # 移除 batch 维度
        hs_tensor = hs_tensor.squeeze(0)               # [C, H, W]
        depth_tensor = depth_tensor.squeeze(0)         # [1, H, W] -> [H, W]
        depth_metric_tensor = depth_metric_tensor.squeeze(0)  # [1, H, W] -> [H, W]
        mask_tensor = mask_tensor.squeeze(0)           # [1, H, W] -> [H, W]

        depth_tensor = depth_tensor.squeeze(0)
        depth_metric_tensor = depth_metric_tensor.squeeze(0)
        mask_tensor = mask_tensor.squeeze(0)

        return {
            'id': sample_id,
            'hs_image': hs_tensor,
            'depth_map': depth_tensor,
            'depth_metric': depth_metric_tensor,
            'mask': mask_tensor,
        }