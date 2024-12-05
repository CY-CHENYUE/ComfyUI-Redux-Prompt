# Redux Style with Prompt Control

A ComfyUI custom node that provides fine-grained control over style transfer using Redux style models.

一个 ComfyUI 自定义节点，提供使用 Redux 风格模型进行精细风格迁移控制。

## Features / 功能特点

- Combine text prompts with reference image styles
- Adjustable influence for both prompts and reference images
- Multiple style strength options
- Flexible image processing modes
- Support for masked regions
- Automatic style token balancing

---

- 结合文本提示词和参考图像风格
- 可调节的提示词和参考图像影响力
- 多种风格强度选项
- 灵活的图像处理模式
- 支持蒙版区域
- 自动风格 token 平衡

## Installation / 安装

1. Clone this repository to your ComfyUI custom nodes directory:
   将此仓库克隆到你的 ComfyUI 自定义节点目录：

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/CY-CHENYUE/ComfyUI-Redux-Prompt.git
   ```

2. Restart ComfyUI
   重启 ComfyUI

## Parameters / 参数说明

### Required Inputs / 必需输入
- `conditioning`: Original prompt input / 原始提示词输入
- `style_model`: Redux style model / Redux 风格模型
- `clip_vision`: CLIP vision encoder / CLIP 视觉编码器
- `reference_image`: Style source image / 风格来源图像
- `prompt_influence`: Prompt strength (1.0=normal) / 提示词强度 (1.0=正常)
- `reference_influence`: Image influence (1.0=normal) / 图像影响 (1.0=正常)
- `style_token_reduction`: Style strength options / 风格强度选项
  - `strong style`: Maximum style influence / 最强风格影响
  - `balanced`: Balanced style and prompt / 平衡风格和提示词
  - `weak style`: Minimum style influence / 最弱风格影响
  - `auto balance`: Automatic adjustment / 自动调节
- `reduction_interpolation`: Token reduction method / Token 缩减方法
- `image_processing_mode`: Image processing mode / 图像处理模式
  - `center crop`: Square center crop / 正方形中心裁剪
  - `keep aspect ratio`: Maintain original ratio / 保持原始比例
  - `autocrop with mask`: Automatic crop using mask / 使用蒙版自动裁剪

### Optional Inputs / 可选输入
- `mask`: Optional mask for local control / 用于局部控制的可选蒙版
- `autocrop_padding`: Padding pixels for autocrop (0-256) / 自动裁剪的边距像素 (0-256)

## Usage Example / 使用示例

1. Add the "Redux Style with Prompt Control" node to your workflow
   将 "Redux Style with Prompt Control" 节点添加到你的工作流程中

2. Connect required inputs:
   连接必需的输入：
   - Text prompt conditioning
   - Redux style model
   - CLIP Vision model
   - Reference image

3. Adjust parameters as needed
   根据需要调整参数

4. Connect the output to your image generation pipeline
   将输出连接到你的图像生成管线

## Notes / 注意事项

- Higher `prompt_influence` values will emphasize the text prompt
- Higher `reference_influence` values will emphasize the reference image style
- The `auto balance` mode automatically adjusts style tokens based on prompt length
- Mask input is only used when `autocrop with mask` mode is selected

---

- 较高的 `prompt_influence` 值会强调文本提示词
- 较高的 `reference_influence` 值会强调参考图像风格
- `auto balance` 模式会根据提示词长度自动调整风格 tokens
- 蒙版输入仅在选择 `autocrop with mask` 模式时使用

