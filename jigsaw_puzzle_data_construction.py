import argparse
import os
import json
import random
from tqdm import tqdm
from PIL import Image
import torch

from safetensors.torch import load_file
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

class MaxLongEdgeMinShortEdgeResize(torch.nn.Module):
    """Resize the input image so that its longest side and shortest side are within a specified range,
    ensuring that both sides are divisible by a specified stride.

    Args:
        max_size (int): Maximum size for the longest edge of the image.
        min_size (int): Minimum size for the shortest edge of the image.
        stride (int): Value by which the height and width of the image must be divisible.
        max_pixels (int): Maximum pixels for the full image.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR``, and ``InterpolationMode.BICUBIC`` are supported.
            The corresponding Pillow integer constants, e.g., ``PIL.Image.BILINEAR`` are also accepted.
        antialias (bool, optional): Whether to apply antialiasing (default is True).
    """

    def __init__(
        self, 
        max_size: int, 
        min_size: int, 
        stride: int, 
        max_pixels: int,
        interpolation=InterpolationMode.BICUBIC, 
        antialias=True
    ):
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

    def _make_divisible(self, value, stride):
        """Ensure the value is divisible by the stride."""
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(self, width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = self._make_divisible(new_width, self.stride)
        new_height = self._make_divisible(new_height, self.stride)
        return new_width, new_height

    def forward(self, img, img_num=1):
        """
        Args:
            img (PIL Image): Image to be resized.
            img_num (int): Number of images, used to change max_tokens.
        Returns:
            PIL Image or Tensor: Rescaled image with divisible dimensions.
        """
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)

        # Ensure the number of pixels does not exceed max_pixels
        if new_width * new_height > self.max_pixels / img_num:
            scale = self.max_pixels / img_num / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        # Ensure longest edge does not exceed max_size
        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        return F.resize(img, (new_height, new_width), self.interpolation, antialias=self.antialias)


class ImageTransform:
    def __init__(
        self, 
        max_image_size, 
        min_image_size, 
        image_stride, 
        max_pixels=14*14*9*1024,
        image_mean=[0.5, 0.5, 0.5], 
        image_std=[0.5, 0.5, 0.5]
    ):
        self.stride = image_stride

        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size, 
            min_size=min_image_size, 
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=image_mean, std=image_std, inplace=True)

    def __call__(self, img, img_num=1):
        img = self.resize_transform(img, img_num=img_num)
        img = self.to_tensor_transform(img)
        img = self.normalize_transform(img)
        return img


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image

def build_transform():
    max_image_size = 154
    min_image_size = 126
    image_stride = 14
    max_pixels = 2007040

    image_transform = ImageTransform(
        max_image_size=max_image_size,
        min_image_size=min_image_size,
        image_stride=image_stride,
        max_pixels=max_pixels,
    )

    return image_transform

def generate_puzzle_prompt(n):
    input_string_1 = '''
You will solve an N×N image jigsaw reconstruction task.

The original image was cut into {K} = {N}×{N} tiles. I will provide the tiles in a fixed order, and each tile has a fixed ID.
All tiles keep their original orientation — no rotation and no flipping are allowed.

Tiles (IDs are fixed)

I will provide {K} tiles as:
'''.format(N=n, K=n*n)

    input_string_3 = '''
Task

Determine the correct placement of all tiles in an {N}×{N} grid so that they form a coherent and visually consistent original image.

Output rules (strict)

Output only a sequence of {K} integers representing tile IDs placed in row-major order (left to right, top to bottom):
Row 1: (Top-Left → Top-Right), ..., until Row {N}.

Use each tile ID from 1 to {K} exactly once.

Output no explanations, no reasoning, no additional text, and no punctuation other than spaces.

Output format example (format only)

'''.format(N=n, K=n*n)
    
    if n == 2:
        input_string_2 = '''
Tile 1: <image>

Tile 2: <image>

Tile 3: <image>

Tile 4: <image>
'''
    elif n == 3:
        input_string_2 = '''
Tile 1: <image>

Tile 2: <image>

Tile 3: <image>

Tile 4: <image>

Tile 5: <image>

Tile 6: <image>

Tile 7: <image>

Tile 8: <image>

Tile 9: <image>
'''
    else:
        assert n == 4
        input_string_2 = '''
Tile 1: <image>

Tile 2: <image>

Tile 3: <image>

Tile 4: <image>

Tile 5: <image>

Tile 6: <image>

Tile 7: <image>

Tile 8: <image>

Tile 9: <image>

Tile 10: <image>

Tile 11: <image>

Tile 12: <image>

Tile 13: <image>

Tile 14: <image>

Tile 15: <image>

Tile 16: <image>
'''

    if n == 2:
        input_string_4 = '''
a1 a2
a3 a4
'''
    elif n == 3:
        input_string_4 = '''
a1 a2 a3
a4 a5 a6
a7 a8 a9
'''
    else:
        assert n == 4
        input_string_4 = '''
a1 a2 a3 a4
a5 a6 a7 a8
a9 a10 a11 a12
a13 a14 a15 a16
'''
    return input_string_1 + input_string_2 + input_string_3 + input_string_4


def convert(input_dir, output_dir):
    input_files = os.listdir(input_dir)

    all_samples = []
    
    for input_file in tqdm(input_files):
        with open(os.path.join(input_dir, input_file), "r", encoding="utf-8") as f:
            data_item = json.load(f)
        original_img_path = data_item["original_image_path"]
        processing_steps = data_item["processing_steps"]
        puzzle_info = data_item["puzzle"]
        puzzle_size = puzzle_info["size"]
        shuffled_order = puzzle_info["shuffled_order"]
        img = Image.open(original_img_path)
        img = img.convert("RGB")  # 初始转为RGB，和预处理一致
        for step in processing_steps:
            step_type = step["step_type"]
            if step_type == "crop":
                crop_bbox = step["crop_bbox"]
                img = img.crop(tuple(crop_bbox))
            elif step_type == "resize":
                resized_size = step["resized_size"]
                img = img.resize(resized_size, Image.Resampling.LANCZOS)
        
        img_w, img_h = img.size
        assert img_w == img_h
        if puzzle_size == 2:
            assert img_w == 140*2, f"{input_file} {img_w}"
        elif puzzle_size == 3:
            assert img_w == 140*3, f"{input_file} {img_w}"
        else:
            assert img_w == 140*4, f"{input_file} {img_w}"

        # 生成原始顺序的子图列表（idx: 0~n²-1）
        piece_size = 140
        original_pieces = []
        for idx in range(puzzle_size * puzzle_size):
            row = idx // puzzle_size
            col = idx % puzzle_size
            left = col * piece_size
            top = row * piece_size
            right = left + piece_size
            bottom = top + piece_size
            piece = img.crop((left, top, right, bottom))
            piece_rgb = pil_img2rgb(piece)
            original_pieces.append(piece_rgb)

        # 4. 按shuffled_order调整子图顺序，生成最终raw_images
        # shuffled_order中每个元素是"原始idx"，对应original_pieces的索引
        raw_images = [original_pieces[idx] for idx in shuffled_order]

        # 处理文本
        puzzle_prompt = generate_puzzle_prompt(puzzle_size)
        
        # 正确答案
        inverse_map = {original_idx: shuffled_pos for shuffled_pos, original_idx in enumerate(shuffled_order)}
        correct_order = list(range(puzzle_size * puzzle_size))  # 正确顺序：0,1,2,...n²-1
        answer = [inverse_map[idx] + 1 for idx in correct_order]  # +1转为1开始
        matrix = [answer[i*puzzle_size : (i+1)*puzzle_size] for i in range(puzzle_size)]
        matrix_str = "\n".join([" ".join(map(str, row)) for row in matrix])

        sample_save_path = os.path.join(output_dir, input_file.split(".")[0])
        os.makedirs(sample_save_path, exist_ok=True)

        # ========== 补充：保存子图 ==========
        image_paths = []
        for idx, piece in enumerate(raw_images):
            img_filename = f"{idx+1:02d}.png"
            img_save_path = os.path.join(sample_save_path, img_filename)
            piece.save(img_save_path, format='PNG')
            abs_img_path = os.path.abspath(img_save_path)
            image_paths.append(abs_img_path)

        # ========== 补充：生成单样本JSON ==========
        sample_json = {
            "messages": [
                {
                    "content": puzzle_prompt,
                    "role": "user"
                },
                {
                    "content": matrix_str,
                    "role": "assistant"
                }
            ],
            "images": image_paths  # 填入所有子图的绝对路径
        }
        # 保存单样本JSON文件
        json_filename = f"{input_file.split('.')[0]}.json"
        json_save_path = os.path.join(sample_save_path, json_filename)
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(sample_json, f, ensure_ascii=False, indent=2)

        # ========== 补充：添加到全局样本列表 ==========
        all_samples.append(sample_json)
    
    with open(os.path.join(output_dir, "all_samples.json"), 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    image_transform = build_transform()
    convert("/dataHW/workspace/fengningya/Bagel/puzzle_dataset/test", "/dataHW/workspace/fengningya/LlamaFactory/puzzle_dataset/test")
    convert("/dataHW/workspace/fengningya/Bagel/puzzle_dataset/train", "/dataHW/workspace/fengningya/LlamaFactory/puzzle_dataset/train")
    convert("/dataHW/workspace/fengningya/Bagel/puzzle_dataset/val", "/dataHW/workspace/fengningya/LlamaFactory/puzzle_dataset/val")
    