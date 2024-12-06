import numpy as np
import torch
import comfy
import folder_paths
import nodes

class ReduxPromptStyler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # 原始的提示词条件输入
                "conditioning": ("CONDITIONING", ),
                # Redux 风格模型
                "style_model": ("STYLE_MODEL", ),
                # CLIP 视觉编码器
                "clip_vision": ("CLIP_VISION", ),
                # 参考图像输入
                "reference_image": ("IMAGE",),
                
                # 控制提示词的影响强度 (1.0=正常强度, <1.0减弱, >1.0增强)
                "prompt_influence": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                
                # 控制参考图像的影响强度 (1.0=正常强度, <1.0减弱, >1.0增强)
                "reference_influence": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                
                # 控制风格tokens的网格大小
                "style_grid_size": ("INT", {
                    "default": 9,            # 默认值改为9 (对应5x5网格)
                    "min": 1,                # 第1个选项
                    "max": 14,               # 第14个选项
                    "step": 1,               # 每次调整1步
                    "display": "slider",     # 显示为滑块
                }),
                
                # 插值方法择
                "interpolation_mode": (["bicubic", "bilinear", "nearest", "area"], {
                    "default": "bicubic",
                }),
                
                # 参考图像的处理模式
                "image_processing_mode": ([
                    "center crop (square)",      # 中心裁剪为正方形
                    "keep aspect ratio",         # 保持原始宽高比
                    "autocrop with mask"         # 根据蒙版自动裁剪
                ], {
                    "default": "center crop (square)"
                }),
            },
            "optional": {
                # 可选的蒙版输入，用于局部控制或自动裁剪
                "mask": ("MASK", ),
                
                # 使用自动裁剪时的边距像素值
                "autocrop_padding": ("INT", {
                    "default": 8,        # 默认32像素边距
                    "min": 0,            # 无边距
                    "max": 256,          # 最大256像素边距
                    "step": 8,
                    "display_step": 8,
                    "display": "slider"
                })
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "IMAGE")
    FUNCTION = "apply_style_with_prompt"
    CATEGORY = "reduxPrompt"

    # 添加节点说明
    DESCRIPTION = """Redux Style with Prompt Control

[English]
- conditioning: Original prompt input
- style_model: Redux style model
- clip_vision: CLIP vision encoder
- reference_image: Style source image
- prompt_influence: Prompt strength (1.0=normal)
- reference_influence: Image influence (1.0=normal)
- style_grid_size: Style detail level (1=strongest 27x27, 14=weakest 1x1)
- interpolation_mode: Token interpolation method
- image_processing_mode: Image mode (crop/aspect/mask)
- mask: Optional mask
- autocrop_padding: Cropping padding (0-256)

[中文]
- conditioning: 原始提示词输入
- style_model: Redux 风格模型
- clip_vision: CLIP 视觉编码器
- reference_image: 风格来源图像
- prompt_influence: 提示词强度 (1.0=正常)
- reference_influence: 图像影响 (1.0=正常)
- style_grid_size: 风格细节等级 (1=最强 27x27, 14=最弱 1x1)
- interpolation_mode: Token插值方法
- image_processing_mode: 图像模式 (裁剪/比例/蒙版)
- mask: 可选蒙版
- autocrop_padding: 裁剪边距 (0-256)"""

    def prepare_image(self, image, mask=None, mode="center crop (square)", 
                     padding=32, desired_size=384):
        """
        预处理参考图像到指定大小和格式
        
        Args:
            image: 输入图像 tensor [B, H, W, C]
            mask: 可选的蒙版 tensor
            mode: 图像处理模式
            padding: 自动裁剪时的边距像素值
            desired_size: 目标图像大小 (默认 384x384)
        
        Returns:
            处理后的图像和蒙版
        """
        B, H, W, C = image.shape
        
        if mode == "center crop (square)":
            # 计算裁剪位置（居中裁剪）
            crop_size = min(H, W)
            x = max(0, (W - crop_size) // 2)
            y = max(0, (H - crop_size) // 2)
            
            # 执行裁剪操作，确保不超出边界
            end_x = x + crop_size
            end_y = y + crop_size
            image = image[:, y:end_y, x:end_x, :]
            
            # 调整图像大小到 desired_size
            image = torch.nn.functional.interpolate(
                image.transpose(-1, 1),
                size=(desired_size, desired_size),
                mode="bicubic",
                antialias=True,
                align_corners=True
            ).transpose(1, -1)
        
        elif mode == "keep aspect ratio":
            # Resize while keeping aspect ratio
            image = torch.nn.functional.interpolate(
                image.transpose(-1, 1),
                size=(W, H),
                mode="bicubic",
                antialias=True,
                align_corners=True
            ).transpose(1, -1)
        
        elif mode == "autocrop with mask" and mask is not None:
            # 获取mask中非零区域的边界框坐标
            mask_np = mask.squeeze(0).cpu().numpy()
            nonzero_indices = np.nonzero(mask_np)
            if len(nonzero_indices[0]) == 0:
                # 如果mask为空，回退到center crop模式
                return self.prepare_image(image, mode="center crop (square)", 
                                       desired_size=desired_size)
            
            # 获取边界框坐标
            min_y, max_y = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
            min_x, max_x = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
            
            # 计算mask区域的宽度和高度
            mask_width = max_x - min_x + 1
            mask_height = max_y - min_y + 1
            
            # 直接使用像素值作为padding
            padding_x = padding
            padding_y = padding
            
            # 计算目标尺寸
            target_width = mask_width + (2 * padding_x)
            target_height = mask_height + (2 * padding_y)
            
            # 确保尺寸是8的倍数
            target_width = ((target_width + 7) // 8) * 8
            target_height = ((target_height + 7) // 8) * 8
            
            # 计算中心点
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            
            # 计算裁剪区域
            crop_x = center_x - (target_width // 2)
            crop_y = center_y - (target_height // 2)
            
            # 确保裁剪区域在图像范围内
            crop_x = max(0, min(crop_x, W - target_width))
            crop_y = max(0, min(crop_y, H - target_height))
            
            # 执行裁剪
            image = image[:, crop_y:crop_y+target_height, crop_x:crop_x+target_width, :]
            
            # 调整到目标尺寸，保持宽高比
            aspect_ratio = target_width / target_height
            if aspect_ratio > 1:  # 宽大于高
                new_height = desired_size
                new_width = int(desired_size * aspect_ratio)
            else:  # 高大于宽
                new_width = desired_size
                new_height = int(desired_size / aspect_ratio)
            
            # 确保尺寸是8的倍数
            new_width = ((new_width + 7) // 8) * 8
            new_height = ((new_height + 7) // 8) * 8
            
            # 调整图像大小 - 修正通道顺序
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            image = torch.nn.functional.interpolate(
                image,
                size=(new_height, new_width),
                mode="bicubic",
                antialias=True,
                align_corners=True
            )
            image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        
        return image



    def apply_style_with_prompt(self, clip_vision, reference_image, style_model, conditioning, 
                              prompt_influence, reference_influence, style_grid_size,
                              interpolation_mode, image_processing_mode,
                              mask=None, autocrop_padding=32):
        """
        将参考图像的风格应用到提示词条件中
        
        Args:
            clip_vision: CLIP视觉编码器
            reference_image: 参考图像
            style_model: Redux风格模型
            conditioning: 原始提示词条件
            prompt_influence: 提示词影响强度
            reference_influence: 参考图影响强度
            style_grid_size: 风格token网格大小
            interpolation_mode: token缩减插值法
            image_processing_mode: 图像处理模式
            mask: 可选的蒙版
            autocrop_padding: 自动裁剪边距
        
        Returns:
            tuple: (处理后的条, 处理后图像)
        """
        # 将文字选项转换为对应的数值
        reduction_map = {
            "strongest style (27x27)": 1,
            "strong style (9x9)": 3,
            "balanced style (5x5)": 5,
            "weak style (3x3)": 9,
            "weakest style (1x1)": 27,
            "auto balance": None  # 使用 None 表示自动平衡逻辑
        }
        target_size = style_grid_size
        reduction_factor = target_size
        
        # 预处理参考图像到指定大小和格式
        processed_image = self.prepare_image(
            reference_image,
            mask=mask,  # 传入蒙版参数
            mode=image_processing_mode,  # 使用函数参数中的图像处理模式
            padding=autocrop_padding,  # 使用函数参数中的自动裁剪边距
            desired_size=384  # 可以根据需要修改目尺寸
        )
        
        # 获取 CLIP 视觉输出
        clip_vision_output = clip_vision.encode_image(processed_image)
        
        # 获取风格条件 (27x27 patches)
        cond = style_model.get_cond(clip_vision_output)
        cond = cond.flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        
        # 获取提示词的 tokens 数量
        prompt_tokens = conditioning[0][0].shape[1]  # 获取第一个条件的 token 数量

        # 计算目标网格大小（从滑块值转换到实际的网格大小）
        grid_sizes = [27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1]
        target_size = grid_sizes[style_grid_size - 1]  # 因为滑块值从1开始
        
        # 应用下采样来减少 tokens 数量
        b, t, h = cond.shape
        m = int(np.sqrt(t))
        cond = cond.view(b, m, m, h)
        
        if target_size < m:  # 只在需要降采样时处理
            cond = torch.nn.functional.interpolate(
                cond.transpose(1, -1),
                size=(target_size, target_size),
                mode=interpolation_mode,
                align_corners=True if interpolation_mode in ["bicubic", "bilinear"] else None
            )
            cond = cond.transpose(1, -1).reshape(b, -1, h)
        
        # 应用风格权重
        cond = cond * (reference_influence * reference_influence)
        
        # 合���条件
        c = []
        for t in conditioning:
            # 先应用提示词权重
            prompt_cond = t[0] * (prompt_influence * prompt_influence)
            
            # 合并条件，确保提示词在前，风格条件在后
            combined_cond = torch.cat((prompt_cond, cond), dim=1)
            
            # 保持原始的 cross attention 和其他参数
            n = [combined_cond, t[1].copy()]
            c.append(n)
            
        return (c, processed_image)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "ReduxPromptStyler": ReduxPromptStyler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReduxPromptStyler": "Redux Style with Prompt Control"
}