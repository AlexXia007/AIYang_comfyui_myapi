"""
Banana2 ComfyUI Node
支持文生图、图生图、多图生图的异步并发API调用
"""

import asyncio
import json
import time
import requests
from typing import Dict, List, Tuple, Optional, Any
import torch
import numpy as np
from PIL import Image
import io
import base64


def _calculate_aspect_ratio(width: int, height: int) -> str:
    """计算图片的宽高比，返回最接近的NanoBanana支持的比例"""
    # 验证输入参数
    if width is None or height is None or width <= 0 or height <= 0:
        print(f"[警告] 无效的图片尺寸 width={width}, height={height}，使用默认比例 1:1")
        return "1:1"

    ratio = width / height
    print(f"[调试] 计算宽高比: {width}/{height} = {ratio:.6f}")

    # 定义NanoBanana支持的比例及其阈值
    supported_ratios = {
        "1:1": 1.0,        # 正方形
        "9:16": 9/16,      # 0.5625 (竖屏手机)
        "16:9": 16/9,      # 1.777... (横屏宽屏)
        "3:4": 3/4,        # 0.75 (竖屏)
        "4:3": 4/3,        # 1.333... (横屏)
        "3:2": 3/2,        # 1.5 (横屏)
        "2:3": 2/3,        # 0.666... (竖屏)
        "5:4": 5/4,        # 1.25 (横屏)
        "4:5": 4/5,        # 0.8 (竖屏)
        "21:9": 21/9,      # 2.333... (超宽屏)
    }

    # 找到差值最小的比例
    min_diff = float('inf')
    best_ratio = "1:1"  # 默认值

    for ratio_name, target_ratio in supported_ratios.items():
        diff = abs(ratio - target_ratio)
        print(f"[调试] 比例 {ratio_name} ({target_ratio:.6f}): 差值 = {diff:.6f}")
        if diff < min_diff:
            min_diff = diff
            best_ratio = ratio_name

    print(f"[调试] 最终匹配比例: {best_ratio} (差值 = {min_diff:.6f})")
    return best_ratio


def _get_image_size_with_exif(image: Image.Image) -> Tuple[int, int]:
    """获取图片的实际尺寸，考虑EXIF方向信息

    当图片有EXIF方向信息（orientation）时，需要根据方向信息调整宽高。
    例如：如果orientation=6（顺时针旋转90度），则实际显示时需要交换宽高。

    Args:
        image: PIL Image对象

    Returns:
        (width, height): 实际显示的尺寸
    """
    width, height = image.size

    # 检查EXIF方向信息
    try:
        exif = image.getexif()
        orientation = exif.get(274)  # EXIF标签274是Orientation
        if orientation:
            # orientation值说明：
            # 1 = 正常（0度）- 不需要交换
            # 3 = 旋转180度 - 不需要交换（尺寸不变）
            # 6 = 顺时针旋转90度（需要交换宽高）
            # 8 = 逆时针旋转90度（需要交换宽高）
            if orientation in [6, 8]:  # 需要旋转90度或270度
                # 交换宽高
                width, height = height, width
    except Exception:
        # 如果获取EXIF失败或图片没有EXIF信息，使用原始尺寸（已赋值，无需修改）
        pass

    return width, height


class Banana2BatchNode:
    """
    Banana2 ComfyUI节点 - 支持并发多组任务处理
    """
    CATEGORY = "AIYang007_myapi"

    @classmethod
    def INPUT_TYPES(cls):
        """定义输入插槽（按组顺序：image_#.1..image_#.4, prompt_#；prompt为插槽-only）"""
        required = {
            "provider": ("STRING", {
                "default": "comfly",
                "tooltip": "供应商名称"
            }),
            "base_url": ("STRING", {
                "default": "https://ai.comfly.chat",
                "tooltip": "API基础地址"
            }),
            "api_key": ("STRING", {
                "tooltip": "API密钥"
            }),
            "model": (["nano-banana-2", "nano-banana-2-2k", "nano-banana-2-4k", "nano-banana"], {
                "default": "nano-banana-2-2k",
                "tooltip": "nano-banana系列模型"
            }),
            "mode": (["Text2Img", "Img2Img"], {
                "default": "Img2Img",
                "tooltip": "图像生成模式"
            }),
            "aspect_ratio": (["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {
                "default": "auto",
                "tooltip": "图像宽高比 (auto=根据输入图片自动计算)"
            }),
            "response_format": (["url", "b64_json"], {
                "default": "url",
                "tooltip": "响应格式"
            }),
            "mode": (["Text2Img", "Img2Img"], {
                "default": "Img2Img",
                "tooltip": "图像生成模式"
            }),
            "img_size": (["1K", "2K", "4K"], {
                "default": "2K",
                "tooltip": "图片尺寸"
            }),
            "img_n": ("INT", {
                "default": 1,
                "min": 1,
                "max": 1,
                "tooltip": "生成图片数量 (只能填1)"
            }),
            "seed": ("INT", {
                "default": 0,
                "min": 0,
                "max": 0xffffffffffffffff,
                "tooltip": "随机种子值，每次点击重新生成随机符合comfyui规范的种子值"
            }),
            "timeout": ("INT", {
                "default": 200,
                "min": 10,
                "max": 600,
                "tooltip": "每一次请求超时(秒) ，如果超时不管是否返回结果，立即判定超时"
            }),
            "retry_count": ("INT", {
                "default": 0,
                "min": 0,
                "max": 5,
                "tooltip": "每一个请求如果失败后的再次重试次数"
            }),
            "node_enabled": ("BOOLEAN", {
                "default": True,
                "tooltip": "节点开关 若为关程序不执行跳过(视为成功执行)"
            })
        }

        # 可选的组输入（可以为None）
        optional = {}
        for group in range(1, 11):
            for img_idx in range(1, 5):
                optional[f"image_{group}.{img_idx}"] = ("IMAGE", {
                    "tooltip": f"组{group}的第{img_idx}张参考图像"
                })

            # prompt 仅作为插槽，不在前端显示文本输入框；使用 forceInput=True 强制仅插槽模式
            optional[f"prompt_{group}"] = ("STRING", {
                "tooltip": f"组{group}的文本提示词（仅插槽）",
                "forceInput": True
            })

        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING",  # 合并输出: images, urls, responses
                   "IMAGE", "STRING", "INT", "STRING",    # group1: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group2: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group3: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group4: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group5: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group6: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group7: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group8: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group9: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group10: image, url, response, info
                   "STRING")                              # stats

    RETURN_NAMES = ("images", "urls", "responses",  # 合并输出
                   "group1_image", "group1_url", "group1_response", "group1_info",  # group1
                   "group2_image", "group2_url", "group2_response", "group2_info",  # group2
                   "group3_image", "group3_url", "group3_response", "group3_info",  # group3
                   "group4_image", "group4_url", "group4_response", "group4_info",  # group4
                   "group5_image", "group5_url", "group5_response", "group5_info",  # group5
                   "group6_image", "group6_url", "group6_response", "group6_info",  # group6
                   "group7_image", "group7_url", "group7_response", "group7_info",  # group7
                   "group8_image", "group8_url", "group8_response", "group8_info",  # group8
                   "group9_image", "group9_url", "group9_response", "group9_info",  # group9
                   "group10_image", "group10_url", "group10_response", "group10_info", # group10
                   "stats")  # 统计

    FUNCTION = "execute"
    OUTPUT_NODE = False

    def __init__(self):
        self.session = requests.Session()

    def execute(self, **kwargs):
        """主执行方法"""
        # 检查节点是否启用
        if not kwargs.get("node_enabled", True):
            return self._get_empty_outputs()

        try:
            # ===== 调试信息: 输入参数详情 =====
            print("\n[DEBUG] Banana2Node 执行开始 =====")
            print(f"[INFO] 节点启用状态: {kwargs.get('node_enabled', True)}")
            print(f"[INFO] 基础URL: {kwargs.get('base_url', 'N/A')}")
            print(f"[INFO] API密钥: {'已配置' if kwargs.get('api_key') else '未配置'}")
            print(f"[INFO] 模型: {kwargs.get('model', 'N/A')}")
            print(f"[INFO] 模式: {kwargs.get('mode', 'N/A')}")
            print(f"[INFO] 宽高比: {kwargs.get('aspect_ratio', 'N/A')}")
            print(f"[INFO] 图片尺寸: {kwargs.get('img_size', 'N/A')}")
            print(f"[INFO] 图片数量: {kwargs.get('img_n', 'N/A')}")
            print(f"[INFO] 种子: {kwargs.get('seed', 'N/A')}")
            print(f"[INFO] 响应格式: {kwargs.get('response_format', 'N/A')}")
            print(f"[INFO] 水印: {kwargs.get('watermark', 'N/A')}")
            print(f"[INFO] 流式输出: {kwargs.get('stream', 'N/A')}")
            print(f"[INFO] 并发数: {kwargs.get('concurrency', 'N/A')}")
            print(f"[INFO] 超时时间: {kwargs.get('timeout', 'N/A')}")
            print(f"[INFO] 重试次数: {kwargs.get('retry_count', 'N/A')}")

            # 显示各组的输入状态
            print("\n[DEBUG] 各组输入状态:")
            for group in range(1, 11):
                has_images = any(kwargs.get(f"image_{group}.{i}") is not None for i in range(1, 5))
                prompt = kwargs.get(f"prompt_{group}")
                print(f"  组{group}: 图片={has_images}, 提示词={'有' if prompt else '无'}")

            # 解析输入参数
            config = self._parse_config(kwargs)
            tasks = self._parse_tasks(kwargs, config)

            print(f"\n📊 解析结果: 共{len(tasks)}个任务, 其中{len([t for t in tasks if t['is_valid']])}个有效")
            print("=" * 50)

            # 过滤有效任务
            valid_tasks = [task for task in tasks if task["is_valid"]]

            if not valid_tasks:
                print("Banana2: 没有有效的任务组")
                return self._get_empty_outputs()

            # 执行任务
            try:
                # 首先尝试使用asyncio.run() (推荐方式)
                results = asyncio.run(self._execute_tasks_async(valid_tasks, config))
            except RuntimeError as e:
                # 如果已经有运行中的循环，使用线程执行
                import concurrent.futures
                import threading

                def run_async():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(self._execute_tasks_async(valid_tasks, config))
                    finally:
                        loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    results = future.result()

            # ===== 调试信息: 执行结果详情 =====
            print("\n[DEBUG] Banana2Node 执行结果汇总:")
            print(f"  [INFO] 总任务数: {len(valid_tasks)}")
            print(f"  [SUCCESS] 成功任务: {len([r for r in results if r.get('success', False)])}")
            print(f"  [ERROR] 失败任务: {len([r for r in results if not r.get('success', False)])}")

            for i, result in enumerate(results, 1):
                status = "[SUCCESS]" if result.get("success", False) else "[ERROR]"
                print(f"  任务{i}: {status} {result.get('info', '无信息')}")

            print("\n[DEBUG] 准备返回最终输出...")
            print("=" * 50)

            # 处理结果
            return self._process_results(results)

        except Exception as e:
            print(f"Banana2: 执行出错 - {str(e)}")
            return self._get_empty_outputs()

    def _parse_config(self, kwargs) -> Dict[str, Any]:
        """解析配置参数"""
        config = {
            "provider": kwargs.get("provider", "comfly"),
            "base_url": kwargs.get("base_url", "https://ai.comfly.chat"),
            "api_key": kwargs.get("api_key", ""),
            "model": kwargs.get("model", "nano-banana-2-2k"),
            "mode": kwargs.get("mode", "Img2Img"),
            "aspect_ratio": kwargs.get("aspect_ratio", "auto"),
            "response_format": kwargs.get("response_format", "url"),
            "img_size": kwargs.get("img_size", "2K"),
            "img_n": kwargs.get("img_n", 1),
            "seed": kwargs.get("seed", 0),
            "timeout": kwargs.get("timeout", 200),
            "retry_count": kwargs.get("retry_count", 0),
            "node_enabled": kwargs.get("node_enabled", True)
        }

        # 调试输出配置
        print(f"[DEBUG] 配置解析结果: {config}")
        return config

    def _parse_tasks(self, kwargs, config) -> List[Dict[str, Any]]:
        """解析任务输入"""
        tasks = []
        for group in range(1, 11):
            images = []
            # 在Text2Img模式下，不解析图片输入
            if config["mode"] == "Img2Img":
                for img_idx in range(1, 5):
                    img_key = f"image_{group}.{img_idx}"
                    img = kwargs.get(img_key)
                    if img is not None and not self._is_empty_tensor(img):
                        images.append(self._tensor_to_pil(img))

            prompt = kwargs.get(f"prompt_{group}", "").strip()

            tasks.append({
                "group_id": group,
                "images": images,
                "prompt": prompt,
                "is_valid": self._is_task_valid(images, prompt, config["mode"])
            })

        return tasks

    def _is_empty_tensor(self, tensor: torch.Tensor) -> bool:
        """判断是否为空tensor"""
        if tensor is None:
            return True

        # 检查tensor是否全为0或非常小
        return torch.allclose(tensor, torch.zeros_like(tensor), atol=1e-6)

    def _is_task_valid(self, images: List[Image.Image], prompt: str, mode: str) -> bool:
        """判断任务是否有效"""
        # 执行条件:
        # 文生图模式：该组prompt插槽(prompt_x)为空时候，该组任务不执行API任务（忽略图像输入）
        # 图生图模式：当某一组的四个图像插槽(image_x.1~image_x.4)传入均为空值 或 该组prompt插槽(prompt_x)为空，两个条件满足其中一个时候，该组任务不执行API任务
        # 空值判断：图像为None或空tensor，文本为None或空字符串

        has_valid_images = len(images) > 0
        has_valid_prompt = bool(prompt)

        if mode == "Text2Img":
            # 文生图模式：只有prompt为空时才无效（忽略图像输入）
            return has_valid_prompt
        else:  # Img2Img
            # 图生图模式：图片和prompt都必须有效（同时满足）
            return has_valid_images and has_valid_prompt

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """将ComfyUI图像tensor转换为PIL图像"""
        # ComfyUI图像tensor格式: [B, H, W, C], RGB, 0-1范围
        if tensor.dim() == 4:  # 批次维度
            tensor = tensor[0]  # 取第一张

        # 转换为numpy并缩放到0-255
        np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)

        # 转换为PIL图像
        return Image.fromarray(np_img)

    async def _execute_tasks_async(self, tasks: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """异步执行所有任务"""
        # 根据任务数量动态调整并发数，最大5个并发
        max_concurrent = min(len(tasks), 5)
        semaphore = asyncio.Semaphore(max_concurrent)
        print(f"Banana2: 开始执行 {len(tasks)} 个任务，使用 {max_concurrent} 个并发")

        async def execute_single_task(task):
            async with semaphore:
                return await self._execute_single_task_with_retry(task, config)

        # 并发执行所有任务
        results = await asyncio.gather(*[execute_single_task(task) for task in tasks], return_exceptions=True)

        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Banana2: 任务{tasks[i]['group_id']}执行异常 - {str(result)}")
                processed_results.append({
                    "group_id": tasks[i]["group_id"],
                    "success": False,
                    "image": None,
                    "url": "",
                    "response_code": 2,  # 失败
                    "info": json.dumps({
                        "status": "error",
                        "message": f"执行异常: {str(result)}"
                    }, ensure_ascii=False)
                })
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_single_task_with_retry(self, task: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个任务（带重试）"""
        retry_count = config["retry_count"]

        for attempt in range(retry_count + 1):
            try:
                result = await self._execute_single_task(task, config)
                if result["success"]:
                    return result
                elif attempt < retry_count:
                    await asyncio.sleep(2)  # 重试间隔
                    continue
                else:
                    return result
            except Exception as e:
                if attempt < retry_count:
                    print(f"Banana2: 任务{task['group_id']}第{attempt+1}次尝试失败 - {str(e)}，准备重试")
                    await asyncio.sleep(2)
                    continue
                else:
                    print(f"Banana2: 任务{task['group_id']}最终失败 - {str(e)}")
                    return {
                        "group_id": task["group_id"],
                        "success": False,
                        "image": None,
                        "url": "",
                        "response_code": 2,
                        "info": json.dumps({
                    "status": "error",
                    "message": f"重试{retry_count}次后仍然失败，最后一次错误: {str(e)}"
                }, ensure_ascii=False)
                    }

    async def _execute_single_task(self, task: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个任务"""
        # 构建API请求
        api_url, headers, payload = self._build_api_request(task, config)

        # ===== 调试信息: API请求详情 =====
        print(f"\n[DEBUG] 任务{task['group_id']} API请求构建:")
        print(f"  [URL] 请求URL: {api_url}")
        print(f"  [HEADERS] 请求头: {headers}")
        print(f"  [PAYLOAD] 请求体: {payload}")
        print(f"  [IMAGES] 参考图片数量: {len(task['images'])}")
        print(f"  [PROMPT] 提示词: {task['prompt'][:100]}{'...' if len(task['prompt']) > 100 else ''}")
        print("-" * 30)

        is_comfly_banana = config["provider"] == "comfly" and config["model"].startswith("nano-banana")

        # 发送请求
        try:
            has_images = len(task["images"]) > 0

            if has_images:
                # 图生图：multipart/form-data
                request_data = payload["data"]
                files = payload["files"]

                # 如果files中有多个同名文件，需要转换为requests期望的格式
                if isinstance(files.get("image"), list):
                    # 转换为requests期望的列表格式
                    files_list = []
                    for file_tuple in files["image"]:
                        files_list.append(("image", file_tuple))
                    files = files_list

                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.post(api_url, headers=headers, data=request_data, files=files, timeout=config["timeout"])
                )
            else:
                # 文生图：application/json
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.post(api_url, headers=headers, json=payload, timeout=config["timeout"])
                )

            if response.status_code == 200:
                result_data = response.json()

                # ===== 调试信息: API响应详情 =====
                print(f"[SUCCESS] 任务{task['group_id']} API响应成功:")
                print(f"  [STATUS] 响应状态码: {response.status_code}")
                print(f"  [RESPONSE] 响应数据: {result_data}")
                print(f"  [MODE] 异步模式: {is_comfly_banana}")
                print("-" * 30)

                # Comfly banana模型使用异步模式
                if is_comfly_banana:
                    return await self._handle_async_response(task["group_id"], result_data, config)
                else:
                    # 其他供应商使用同步模式
                    return self._parse_sync_response(task["group_id"], result_data, config["response_format"])
            else:
                print(f"Banana2: 任务{task['group_id']} API请求失败 - {response.status_code}: {response.text}")
                return {
                    "group_id": task["group_id"],
                    "success": False,
                    "image": None,
                    "url": "",
                    "response_code": 2,
                    "info": json.dumps({
                    "status": "error",
                    "message": f"API请求失败 - {response.status_code}",
                    "response_text": response.text
                }, ensure_ascii=False)
                }

        except requests.exceptions.Timeout:
            print(f"Banana2: 任务{task['group_id']} 请求超时")
            return {
                "group_id": task["group_id"],
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": json.dumps({
                    "status": "error",
                    "message": f"请求超时 ({config['timeout']}秒)"
                }, ensure_ascii=False)
            }
        except Exception as e:
            print(f"Banana2: 任务{task['group_id']} 请求异常 - {str(e)}")
            return {
                "group_id": task["group_id"],
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": json.dumps({
                    "status": "error",
                    "message": f"请求异常: {str(e)}"
                }, ensure_ascii=False)
            }

    def _build_api_request(self, task: Dict[str, Any], config: Dict[str, Any]) -> Tuple[str, Dict[str, str], Any]:
        """构建API请求"""
        base_url = config["base_url"].rstrip("/")
        has_images = len(task["images"]) > 0
        is_comfly_banana = config["provider"] == "comfly" and config["model"].startswith("nano-banana")

        # 处理aspect_ratio的auto模式
        final_aspect_ratio = config["aspect_ratio"]
        if config["aspect_ratio"] == "auto":
            if has_images:
                # 获取第一张图片的尺寸
                first_image = task["images"][0]
                width, height = _get_image_size_with_exif(first_image)
                if width and height:
                    final_aspect_ratio = _calculate_aspect_ratio(width, height)
                    print(f"[AUTO] 根据输入图片({width}x{height})计算比例: {final_aspect_ratio}")
                else:
                    final_aspect_ratio = "1:1"
                    print("[AUTO] 无法获取图片尺寸，使用默认比例: 1:1")
            else:
                final_aspect_ratio = "1:1"
                print("[AUTO] 无输入图片，使用默认比例: 1:1")

        # 根据mode决定是否使用图像
        use_images = has_images and config["mode"] == "Img2Img"

        if use_images:
            # 图生图 - 使用multipart/form-data
            api_url = f"{base_url}/v1/images/edits"
            query_params = ""

            # Comfly banana模型使用异步模式
            if is_comfly_banana:
                query_params = "?async=true"
                api_url += query_params

            headers = {
                "Authorization": f"Bearer {config['api_key']}"
            }

            # 构建multipart/form-data
            files = {}
            data = {
                "model": config["model"],
                "prompt": task["prompt"],
                "response_format": config["response_format"],
                "aspect_ratio": final_aspect_ratio,
                "image_size": config["img_size"]
            }

            # 添加图像文件 - 支持多图
            # 存储为列表，稍后在发送时转换为正确的requests格式
            image_files = []
            for i, img in enumerate(task["images"]):
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)
                image_files.append((f"image_{i+1}.png", buffer, "image/png"))

            files["image"] = image_files

            return api_url, headers, {"data": data, "files": files}
        else:
            # 文生图 - 使用application/json
            api_url = f"{base_url}/v1/images/generations"
            query_params = ""

            # Comfly banana模型使用异步模式
            if is_comfly_banana:
                query_params = "?async=true"
                api_url += query_params

            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": config["model"],
                "prompt": task["prompt"],
                "response_format": config["response_format"],
                "aspect_ratio": final_aspect_ratio,
                "image_size": config["img_size"]
            }

            return api_url, headers, payload

    async def _handle_async_response(self, group_id: int, response_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """处理异步响应，获取task_id并轮询状态"""
        try:
            # 从响应中获取task_id
            task_id = None
            if "task_id" in response_data:
                # 直接在响应根层级
                task_id = response_data["task_id"]
            elif "data" in response_data and isinstance(response_data["data"], dict) and "task_id" in response_data["data"]:
                # 在data子对象中
                task_id = response_data["data"]["task_id"]
            elif "data" in response_data and isinstance(response_data["data"], str):
                # data字段直接是task_id字符串
                task_id = response_data["data"]

            if task_id:
                print(f"Banana2: 任务{group_id} 异步任务已提交，task_id: {task_id}")
                # 开始轮询查询状态
                return await self._poll_task_status(group_id, task_id, config)
            else:
                print(f"Banana2: 任务{group_id} 异步响应中未找到task_id: {response_data}")
                return {
                    "group_id": group_id,
                    "success": False,
                    "image": None,
                    "url": "",
                    "response_code": 2,
                    "info": json.dumps({
                        "status": "error",
                        "message": f"异步响应中未找到task_id",
                        "response_data": response_data
                    }, ensure_ascii=False)
                }

        except Exception as e:
            print(f"Banana2: 任务{group_id} 处理异步响应异常 - {str(e)}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                    "info": json.dumps({
                        "status": "error",
                        "message": f"处理异步响应异常: {str(e)}",
                        "response_data": response_data
                    }, ensure_ascii=False)
            }

    async def _poll_task_status(self, group_id: int, task_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """轮询查询任务状态，每5秒查询一次"""
        base_url = config["base_url"].rstrip("/")
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }

        max_polls = 60  # 最多轮询60次（5分钟）
        poll_count = 0

        while poll_count < max_polls:
            poll_count += 1

            try:
                # 构建查询URL
                query_url = f"{base_url}/v1/images/tasks/{task_id}"

                # 发送查询请求
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.get(query_url, headers=headers, timeout=30)
                )

                if response.status_code == 200:
                    status_data = response.json()

                    if "data" in status_data:
                        task_info = status_data["data"]
                        status = task_info.get("status", "")
                        progress = task_info.get("progress", "0%")

                        print(f"Banana2: 任务{group_id} 状态查询 [{poll_count}] - 状态: {status}, 进度: {progress}")

                        if status == "SUCCESS":
                            # 任务完成，解析结果
                            return self._parse_async_success_response(group_id, task_info, config["response_format"])

                        elif status == "FAILURE":
                            # 任务失败
                            fail_reason = task_info.get("fail_reason", "未知错误")
                            print(f"Banana2: 任务{group_id} 生成失败 - {fail_reason}")
                            return {
                                "group_id": group_id,
                                "success": False,
                                "image": None,
                                "url": "",
                                "response_code": 2,
                                "info": json.dumps(task_info, ensure_ascii=False)
                            }

                        elif status in ["IN_PROGRESS", "NOT_START", "PENDING"]:
                            # 任务进行中，继续等待
                            await asyncio.sleep(5)  # 等待5秒
                            continue

                        else:
                            print(f"Banana2: 任务{group_id} 未知状态: {status}")
                            await asyncio.sleep(5)
                            continue

                    else:
                        print(f"Banana2: 任务{group_id} 状态查询响应格式错误: {status_data}")
                        await asyncio.sleep(5)
                        continue

                else:
                    print(f"Banana2: 任务{group_id} 状态查询失败 - {response.status_code}: {response.text}")
                    await asyncio.sleep(5)
                    continue

            except Exception as e:
                print(f"Banana2: 任务{group_id} 状态查询异常 - {str(e)}")
                await asyncio.sleep(5)
                continue

        # 超时
        print(f"Banana2: 任务{group_id} 查询超时，已等待{max_polls * 5}秒")
        return {
            "group_id": group_id,
            "success": False,
            "image": None,
            "url": "",
            "response_code": 2,
                "info": json.dumps({
                    "status": "error",
                    "message": f"异步查询超时，已等待{max_polls * 5}秒"
                }, ensure_ascii=False)
        }

    def _parse_async_success_response(self, group_id: int, task_info: Dict[str, Any], response_format: str) -> Dict[str, Any]:
        """解析异步成功的响应"""
        try:
            if "data" in task_info and "data" in task_info["data"]:
                image_data = task_info["data"]["data"][0]

                # 提取URL
                image_url = image_data.get("url", "")
                if not image_url:
                    # 尝试b64_json
                    b64_data = image_data.get("b64_json", "")
                    if b64_data:
                        # 将base64转换为图像URL (这里简化处理，实际需要上传到服务器)
                        image_url = f"data:image/png;base64,{b64_data}"

                if image_url:
                    # 根据URL格式决定是否下载图片
                    if image_url.startswith("data:image"):
                        # base64格式，需要下载转换
                        image = self._download_image(image_url)
                        if image:
                            print(f"Banana2: 任务{group_id} 图像生成成功 (Base64)")
                            # 根据response_format决定URL返回值
                            return_url = "b64_ok" if response_format == "b64_json" else image_url
                            return {
                                "group_id": group_id,
                                "success": True,
                                "image": image,
                                "url": return_url,
                                "response_code": 1,
                                "info": json.dumps({
                                    "status": "success",
                                    "message": "图像生成成功",
                                    "format": "base64" if response_format == "b64_json" else "url",
                                    "task_info": task_info
                                }, ensure_ascii=False)
                            }
                    else:
                        # URL格式，直接返回URL，不下载图片
                        print(f"Banana2: 任务{group_id} 图像生成成功 (URL): {image_url}")
                        return {
                            "group_id": group_id,
                            "success": True,
                            "image": None,  # URL格式不下载图片
                            "url": image_url,
                            "response_code": 1,
                            "info": json.dumps({
                                "status": "success",
                                "message": "图像生成成功",
                                "format": "url",
                                "task_info": task_info
                            }, ensure_ascii=False)
                        }

            print(f"Banana2: 任务{group_id} 异步响应解析失败 - {task_info}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": json.dumps({
                    "status": "error",
                    "message": f"异步响应解析失败",
                    "response_data": task_info
                }, ensure_ascii=False)
            }

        except Exception as e:
            print(f"Banana2: 任务{group_id} 异步响应解析异常 - {str(e)}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": json.dumps({
                    "status": "error",
                    "message": f"异步响应解析异常: {str(e)}",
                    "response_data": task_info
                }, ensure_ascii=False)
            }

    def _parse_sync_response(self, group_id: int, response_data: Dict[str, Any], response_format: str) -> Dict[str, Any]:
        """解析同步响应"""
        try:
            if "data" in response_data and len(response_data["data"]) > 0:
                image_data = response_data["data"][0]

                # 提取URL
                image_url = image_data.get("url", "")
                if not image_url:
                    # 尝试b64_json
                    b64_data = image_data.get("b64_json", "")
                    if b64_data:
                        # 将base64转换为图像URL (这里简化处理，实际需要上传到服务器)
                        image_url = f"data:image/png;base64,{b64_data}"

                if image_url:
                    # 根据response_format决定是否下载图片
                    # 如果是URL格式，直接返回URL；如果是base64，需要下载转换
                    if image_url.startswith("data:image"):
                        # base64格式，需要下载转换
                        image = self._download_image(image_url)
                        if image:
                            # 根据response_format决定URL返回值
                            return_url = "b64_ok" if response_format == "b64_json" else image_url
                            return {
                                "group_id": group_id,
                                "success": True,
                                "image": image,
                                "url": return_url,
                                "response_code": 1,
                                "info": json.dumps({
                                    "status": "success",
                                    "message": "图像生成成功",
                                    "format": "base64" if response_format == "b64_json" else "url",
                                    "task_info": response_data
                                }, ensure_ascii=False)
                            }
                    else:
                        # URL格式，直接返回URL，不下载图片
                        print(f"[INFO] URL格式响应，直接返回链接: {image_url}")
                        return {
                            "group_id": group_id,
                            "success": True,
                            "image": None,  # URL格式不下载图片
                            "url": image_url,
                            "response_code": 1,
                            "info": json.dumps({
                                "status": "success",
                                "message": "图像生成成功",
                                "format": "url",
                                "task_info": response_data
                            }, ensure_ascii=False)
                        }

            print(f"Banana2: 任务{group_id} 响应解析失败 - {response_data}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": json.dumps({
                    "status": "error",
                    "message": f"同步响应解析失败",
                    "response_data": response_data
                }, ensure_ascii=False)
            }

        except Exception as e:
            print(f"Banana2: 任务{group_id} 响应解析异常 - {str(e)}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": json.dumps({
                    "status": "error",
                    "message": f"同步响应解析异常: {str(e)}",
                    "response_data": response_data
                }, ensure_ascii=False)
            }

    def _download_image(self, url: str, max_retries: int = 2) -> Optional[Image.Image]:
        """下载图像，带重试和完整性检查"""
        for attempt in range(max_retries + 1):
            try:
                if url.startswith("data:image"):
                    # base64数据 - 简化处理，不使用verify()
                    header, data = url.split(",", 1)
                    img_data = base64.b64decode(data)
                    img_buffer = io.BytesIO(img_data)
                    img = Image.open(img_buffer)

                    # 对于base64数据，如果能成功打开图片，说明数据完整
                    # 不需要额外的verify()验证（verify()会关闭图片对象）
                    print(f"[SUCCESS] Base64图片处理成功")
                    return img

                else:
                    # URL下载
                    response = self.session.get(url, timeout=30)
                    if response.status_code == 200:
                        img_buffer = io.BytesIO(response.content)
                        img = Image.open(img_buffer)

                        # 验证图片完整性
                        img.verify()  # 检查图片是否完整
                        img.close()
                        img_buffer.seek(0)  # 重置buffer位置
                        img = Image.open(img_buffer)  # 重新打开

                        print(f"[SUCCESS] URL图片下载并验证成功，大小: {len(response.content)} bytes")
                        return img
                    else:
                        print(f"[ERROR] 图片下载失败，状态码: {response.status_code}")

            except Exception as e:
                print(f"[ERROR] 图片处理失败 (尝试 {attempt + 1}/{max_retries + 1}): {str(e)}")
                if attempt < max_retries:
                    import time
                    time.sleep(1)  # 等待1秒后重试
                    continue

        print(f"[ERROR] 图片下载失败，已重试 {max_retries + 1} 次")
        return None

    def _process_results(self, results: List[Dict[str, Any]]) -> Tuple:
        """处理结果并返回输出"""
        # 分组结果
        group_results = {}
        for result in results:
            group_results[result["group_id"]] = result

        # 统计信息
        valid_tasks = len(results)
        success_tasks = sum(1 for r in results if r["success"])

        # 合并输出：只包含成功的结果
        successful_images = []
        all_urls = []
        all_responses = []

        # 独立组输出
        group_outputs = []

        for group_id in range(1, 11):
            if group_id in group_results:
                result = group_results[group_id]
                # 合并输出
                all_urls.append(result["url"] if result["url"] else "")
                all_responses.append(result["response_code"])

                # 独立组输出
                if result["success"]:
                    if result["image"]:
                        # 有图片数据（Base64格式），需要转换
                        tensor_image = self._pil_to_tensor(result["image"])
                        if tensor_image is not None:
                            successful_images.append(tensor_image)
                            # 为group输出添加批次维度 [1, H, W, C]
                            group_image = tensor_image.unsqueeze(0)
                            group_outputs.extend([
                                group_image,
                                result["url"],
                                result["response_code"],
                                result.get("info", "成功")
                            ])
                        else:
                            # 图片转换失败，当作失败处理
                            print(f"[ERROR] 任务{group_id} 图片转换失败，使用空图片")
                            empty_image = torch.zeros((1, 64, 64, 3))  # 批次格式: [B, H, W, C]
                            group_outputs.extend([
                                empty_image,
                                result["url"],
                                3,  # 转换失败
                                "图片转换失败"
                            ])
                    else:
                        # 无图片数据但有URL（URL格式），创建占位符图像
                        print(f"[INFO] 任务{group_id} URL格式响应: {result['url']}")
                        # 为URL格式创建一个特殊的占位符图像，表示这是URL链接
                        url_placeholder = torch.full((1, 64, 64, 3), 0.5)  # 批次格式，灰色占位符，0-1范围
                        group_outputs.extend([
                            url_placeholder,
                            result["url"],
                            result["response_code"],
                            result.get("info", "URL格式响应")
                        ])
                else:
                    # 失败情况下的独立组输出
                    # ComfyUI图像格式: [B, H, W, C]，torch.Tensor
                    empty_image = torch.zeros((1, 64, 64, 3))
                    group_outputs.extend([
                        empty_image,
                        result["url"],
                        result["response_code"],
                        result.get("info", "未执行")
                    ])
            else:
                # 未执行的任务
                all_urls.append("")
                all_responses.append(0)
                # ComfyUI图像格式: [B, H, W, C]
                empty_image = torch.zeros((1, 64, 64, 3))
                group_outputs.extend([
                    empty_image,
                    "",
                    0,
                    "未执行的任务"
                ])

        # 合并输出images：堆叠所有成功的图像
        if successful_images:
            # 总是堆叠为批次格式 [B, H, W, C]，即使只有一个图像
            merged_images = torch.stack(successful_images)
            print(f"[DEBUG] 合并图像形状: {merged_images.shape}")
        else:
            # 如果没有成功的图像，创建一个占位符图像
            merged_images = torch.full((1, 64, 64, 3), 0.5)  # 灰色占位符 [B, H, W, C]
            print(f"[DEBUG] 空合并图像形状: {merged_images.shape} (占位符)")

        # urls和responses作为JSON字符串
        urls_json = json.dumps(all_urls, ensure_ascii=False)
        responses_json = json.dumps(all_responses, ensure_ascii=False)

        # 统计输出
        stats = f"(有效任务:{valid_tasks}, 成功任务:{success_tasks})"

        # 返回所有输出：合并输出(3) + 独立组输出(30) + 统计输出(1) = 34个
        return tuple([merged_images, urls_json, responses_json] + group_outputs + [stats])

    def _mask_b64_json(self, data: Any) -> Any:
        """屏蔽API响应中的b64_json内容以避免日志溢出"""
        if isinstance(data, dict):
            masked = {}
            for key, value in data.items():
                if key == "b64_json" and isinstance(value, str) and len(value) > 20:
                    # 只保留前20个字符，并显示数据长度
                    data_length = len(value)
                    masked[key] = f"{value[:20]}...[BASE64_DATA_{data_length}_CHARS]"
                else:
                    masked[key] = self._mask_b64_json(value)
            return masked
        elif isinstance(data, list):
            return [self._mask_b64_json(item) for item in data]
        else:
            return data

    def _pil_to_tensor(self, image: Image.Image) -> Optional[torch.Tensor]:
        """将PIL图像转换为ComfyUI期望的torch.Tensor格式，带错误检查"""
        try:
            if image is None:
                print("[ERROR] 输入图像为空")
                return None

            # 注意：图片已经在_download_image中验证过了，这里不再重复验证
            # 如果图片能到达这里，说明它已经是有效的PIL图像

            # 确保RGB模式
            if image.mode != "RGB":
                print(f"[INFO] 转换图片模式: {image.mode} -> RGB")
                image = image.convert("RGB")

            # 检查图片尺寸
            width, height = image.size
            if width == 0 or height == 0:
                print(f"[ERROR] 图片尺寸无效: {width}x{height}")
                return None

            # 转换为numpy数组，保持0-255范围
            print(f"[INFO] 转换图片尺寸: {width}x{height}")
            np_img = np.array(image)

            # 检查数组形状
            if len(np_img.shape) != 3 or np_img.shape[2] != 3:
                print(f"[ERROR] 图片数组形状异常: {np_img.shape}")
                return None

            # 转换为torch.Tensor，归一化到0-1范围，格式: [H, W, C] (ComfyUI标准格式)
            tensor = torch.from_numpy(np_img.astype(np.float32) / 255.0)

            print(f"[SUCCESS] 图片转换为torch.Tensor成功，形状: {tensor.shape}")
            return tensor

        except Exception as e:
            print(f"[ERROR] 图片转tensor失败: {str(e)}")
            return None

    def _get_empty_outputs(self) -> Tuple:
        """返回空的输出"""
        # ComfyUI图像格式: torch.Tensor [H, W, C]，范围0-1
        empty_image = torch.zeros((64, 64, 3), dtype=torch.float32)

        # 合并输出
        merged_outputs = [empty_image, "[]", "[]"]

        # 独立组输出 (10组 × 4)
        group_outputs = []
        for _ in range(10):
            group_outputs.extend([empty_image, "", 0, "未执行的任务"])

        # 统计输出
        stats_output = ["(有效任务:0, 成功任务:0)"]

        return tuple(merged_outputs + group_outputs + stats_output)


# 节点注册映射
NODE_CLASS_MAPPINGS = {
    "Banana2Batch": Banana2BatchNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Banana2Batch": "AIYang007_Banana2Batch"
}
