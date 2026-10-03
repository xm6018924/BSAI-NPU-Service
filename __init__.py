# -*- coding: utf-8 -*-
"""
BSAI NPU Service —— ComfyUI 插件入口
=================================================
把 Intel AI Boost (NPU) 人脸服务挂载到 ComfyUI server：
- GET  /health        -> {status, device, npu_models}
- POST /detect_face   -> {count, faces:[{score,bbox,kps}]}
- POST /landmark_106  -> 106 点人脸关键点
- POST /recog_r50     -> 512 维人脸特征向量
- POST /gender_age    -> 性别年龄

任何 BSAI 插件自动获得 NPU 能力（无需手动 import）：
    from bsai_npu_client import npu, npu_available
"""

import json
import os
import sys
import threading

import aiohttp
from aiohttp import web
import numpy as np

from .npu_service import NPUService, decode_image_bgr

# ---- BSAI 插件协同 SDK：作为 NPU 人脸检测服务提供方注册能力（失败不拖垮插件） ----
_ORCH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "BSAI-ComfyUI-Orchestrator")
if not os.path.isdir(_ORCH):
    _ORCH = r"G:\BSAI-ComfyUI-intel-XPU-GPU-NPU-aki\ComfyUI\custom_nodes\BSAI-ComfyUI-Orchestrator"
if os.path.isdir(_ORCH) and _ORCH not in sys.path:
    sys.path.insert(0, _ORCH)
try:
    from bsai_orch_client import BSAIOrch
except Exception:
    BSAIOrch = None

try:
    if BSAIOrch is not None:
        BSAIOrch.register(
            name="BSAI-NPU-Service",
            kind="face_detect",                 # 本服务即 NPU 人脸检测能力提供方
            hardware=["npu"],
            endpoint="http://127.0.0.1:8191/detect_face",
            health="http://127.0.0.1:8191/health/ready",
        )
except Exception:
    pass
# 注：本插件是 NPU 推理执行体（OpenVINO 编译模型自管并发），不做消费侧
# allocate；跨进程 NPU 互斥由消费方（FaceRefine 等）在检测时持租约。

WEB_DIRECTORY = "./web" if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")) else None

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

_svc = NPUService()


# ---------------- 全局注入：让任何 BSAI 插件直接 import bsai_npu_client ----------------
def _inject_client():
    """把客户端模块注入 sys.modules，其他插件 from bsai_npu_client import npu 即可用"""
    client_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bsai_npu_client.py")
    if not os.path.exists(client_src):
        return
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("bsai_npu_client", client_src)
        mod = importlib.util.module_from_spec(spec)
        # 让客户端直接引用本插件的 _svc（同进程零开销）
        mod._svc = _svc
        mod._direct = True
        sys.modules["bsai_npu_client"] = mod
        spec.loader.exec_module(mod)
        # exec_module 会重置 _svc，重新绑定
        mod._svc = _svc
        mod._direct = True
    except Exception as e:
        print("[BSAI-NPU-Service] 客户端注入跳过: %s" % e)


_inject_client()


# ---------------- 后台预热 NPU 模型 ----------------
def _warmup():
    try:
        import numpy as np
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        _svc.detect_faces(dummy)
        print("[BSAI-NPU-Service] NPU 模型预热完成 (device=%s)" % _svc.device)
    except Exception as e:
        print("[BSAI-NPU-Service] NPU 预热跳过: %s" % e)


threading.Thread(target=_warmup, daemon=True).start()


# ---------------- 路由处理 ----------------
async def npu_health(_):
    return web.json_response(_svc.status())


async def npu_health_ready(_):
    return web.json_response(_svc.ready())


async def npu_detect_face(req):
    try:
        body = await req.json()
        img = decode_image_bgr(body.get("image", ""))
        faces = _svc.detect_faces(img)
        return web.json_response({"count": len(faces), "faces": faces})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def npu_landmark(req):
    try:
        body = await req.json()
        img = decode_image_bgr(body.get("image", ""))
        box = body.get("bbox")
        if not box:
            return web.json_response({"error": "bbox required"}, status=400)
        pts = _svc.landmark_106(img, box)
        return web.json_response({"count": len(pts), "points": pts})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def npu_recog(req):
    try:
        body = await req.json()
        img = decode_image_bgr(body.get("image", ""))
        box = body.get("bbox")
        if not box:
            return web.json_response({"error": "bbox required"}, status=400)
        emb = _svc.recog_r50(img, box)
        return web.json_response({"dim": len(emb), "embedding": emb})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def npu_gender_age(req):
    try:
        body = await req.json()
        img = decode_image_bgr(body.get("image", ""))
        box = body.get("bbox")
        if not box:
            return web.json_response({"error": "bbox required"}, status=400)
        return web.json_response(_svc.gender_age(img, box))
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def npu_llm_generate(req):
    try:
        body = await req.json()
        r = _svc.llm_generate(
            prompt=body.get("prompt", ""),
            max_new_tokens=body.get("max_new_tokens", 128),
            temperature=body.get("temperature", 0.7),
            top_p=body.get("top_p", 0.9),
            enable_thinking=body.get("enable_thinking", False),
            seed=body.get("seed"),
        )
        return web.json_response({"result": r})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def npu_detect_pose(req):
    try:
        body = await req.json()
        img = decode_image_bgr(body.get("image", ""))
        poses = _svc.detect_pose(img)
        return web.json_response({"count": len(poses), "poses": poses})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def npu_segment(req):
    try:
        body = await req.json()
        img = decode_image_bgr(body.get("image", ""))
        fg, alpha = _svc.segment_foreground(img)
        return web.json_response({
            "fg": _encode_bgr_b64(fg),
            "alpha": _encode_alpha_b64(alpha),
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def npu_depth(req):
    try:
        body = await req.json()
        img = decode_image_bgr(body.get("image", ""))
        gray, meta = _svc.detect_depth(img)
        return web.json_response({
            "depth": _encode_bgr_b64(np.dstack([gray, gray, gray])),
            "meta": meta,
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


def _encode_bgr_b64(image_bgr):
    try:
        import cv2
        ok, buf = cv2.imencode(".png", image_bgr)
        if ok:
            return "data:image/png;base64," + _b64(buf.tobytes())
    except Exception:
        pass
    from PIL import Image
    import io
    pil = Image.fromarray(image_bgr[:, :, ::-1])
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + _b64(buf.getvalue())


def _encode_alpha_b64(alpha):
    import base64
    from PIL import Image
    import io
    img = (alpha * 255).astype("uint8")
    pil = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _b64(b):
    import base64
    return base64.b64encode(b).decode()


# ---------------- 注册路由（容错：极端情况下 instance 不可用仍可加载） ----------------
def _register_routes():
    """把 NPU 服务路由挂到 ComfyUI PromptServer（幂等；instance 未就绪时由节点定义处再调一次）"""
    try:
        import server as _server
        _ps = _server.PromptServer.instance
        if _ps is not None and hasattr(_ps, "routes"):
            routes = _ps.routes
            for method, path, handler in [
                ("GET", "/health", npu_health),
                ("GET", "/health/ready", npu_health_ready),
                ("POST", "/detect_face", npu_detect_face),
                ("POST", "/landmark_106", npu_landmark),
                ("POST", "/recog_r50", npu_recog),
                ("POST", "/gender_age", npu_gender_age),
                ("POST", "/llm_generate", npu_llm_generate),
                ("POST", "/segment_foreground", npu_segment),
                ("POST", "/detect_pose", npu_detect_pose),
                ("POST", "/detect_depth", npu_depth),
            ]:
                # ComfyUI 0.38 (aiohttp) 的 RouteTableDef 用 route(method, path)(handler) 装饰器注册，无 add_route
                try:
                    existing = [r for r in routes
                                if getattr(r, "path", None) == path and getattr(r, "method", None) == method]
                    if not existing:
                        routes.route(method, path)(handler)
                except Exception:
                    try:
                        # 兼容旧 aiohttp：add_route 直挂
                        routes.add_route(method, path, handler)
                    except Exception:
                        pass
            print("[BSAI-NPU-Service] NPU 服务已挂载到 ComfyUI: GET /health POST /detect_face /landmark_106 /recog_r50 /gender_age /llm_generate /segment_foreground /detect_pose (device=%s)" % _svc.device)
            return True
        else:
            print("[BSAI-NPU-Service] PromptServer.instance 未就绪，路由将在节点定义处补注册")
            return False
    except Exception as e:
        print("[BSAI-NPU-Service] 路由注册容错跳过:", e)
        return False


_register_routes()


# ---------------- NPU 能力节点（LLM 决策文本 / RMBG 抠图） ----------------
def _img_to_bgr(image):
    """IMAGE tensor (B,H,W,C) 0-1 RGB -> 首帧 BGR uint8"""
    import torch
    img = image[0].detach().cpu().numpy()
    img = np.clip(img, 0.0, 1.0)
    return (img[:, :, ::-1] * 255.0).astype(np.uint8)


def _bgr_to_img(bgr):
    """BGR uint8 -> IMAGE tensor (1,H,W,C) 0-1 RGB"""
    import torch
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(rgb).unsqueeze(0)


class BSAINPU_LLMGenerate:
    """Qwen3-4B（NPU 对称 INT4）决策文本生成：把提示词/决策文本任务卸载到 NPU"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                                      "default": "请用不超过 30 字总结画面内容，并给出下一步处理建议。"}),
                "max_new_tokens": ("INT", {"default": 128, "min": 8, "max": 1024, "step": 8}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "meta_json")
    FUNCTION = "run"
    CATEGORY = "BSAI/NPU"

    def run(self, prompt, max_new_tokens, temperature, top_p, enable_thinking):
        import json
        try:
            from bsai_npu_client import npu
            r = npu.llm_generate(prompt, max_new_tokens=max_new_tokens,
                                 temperature=temperature, top_p=top_p,
                                 enable_thinking=enable_thinking)
            return (r.get("text", ""), json.dumps(r, ensure_ascii=False))
        except Exception as e:
            return ("[NPU LLM 错误] %s" % e, json.dumps({"error": str(e)}, ensure_ascii=False))


class BSAINPU_SegmentForeground:
    """RMBG-1.4（NPU）前景抠图：输入图 -> 前景图 + 前景掩码（替代失效的 comfyui-rmbg）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("foreground", "mask")
    FUNCTION = "run"
    CATEGORY = "BSAI/NPU"

    def run(self, image):
        import torch
        try:
            from bsai_npu_client import npu
            bgr = _img_to_bgr(image)
            fg, alpha = npu.segment_foreground(bgr)
            out_img = _bgr_to_img(fg)
            mask = torch.from_numpy(alpha.astype(np.float32))
            return (out_img, mask)
        except Exception as e:
            # 失败降级：返回原图 + 全 1 掩码，不拖垮工作流
            import torch
            bgr = _img_to_bgr(image)
            h, w = bgr.shape[:2]
            return (_bgr_to_img(bgr), torch.ones((h, w), dtype=torch.float32))


class BSAINPU_PoseDetect:
    """YOLO11n-pose（NPU）人体检测+17 关键点：输出 JSON 姿态信息（决策文本节点可直接引用）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("pose_json",)
    FUNCTION = "run"
    CATEGORY = "BSAI/NPU"

    def run(self, image):
        import json
        try:
            from bsai_npu_client import npu
            bgr = _img_to_bgr(image)
            poses = npu.detect_pose(bgr)
            return (json.dumps({"count": len(poses), "poses": poses}, ensure_ascii=False),)
        except Exception as e:
            return (json.dumps({"error": str(e), "poses": []}, ensure_ascii=False),)


class BSAINPU_DepthEstimate:
    """Depth-Anything-V2-Small（NPU）单目深度估计：输入图 -> 深度灰度图 + meta JSON"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("depth_image", "meta_json")
    FUNCTION = "run"
    CATEGORY = "BSAI/NPU"

    def run(self, image):
        import json
        try:
            from bsai_npu_client import npu
            bgr = _img_to_bgr(image)
            gray, meta = npu.detect_depth(bgr)
            depth_rgb = np.dstack([gray, gray, gray])
            out_img = _bgr_to_img(depth_rgb)
            return (out_img, json.dumps(meta, ensure_ascii=False))
        except Exception as e:
            bgr = _img_to_bgr(image)
            return (_bgr_to_img(bgr), json.dumps({"error": str(e)}, ensure_ascii=False))


NODE_CLASS_MAPPINGS = {
    "BSAINPU_LLMGenerate": BSAINPU_LLMGenerate,
    "BSAINPU_SegmentForeground": BSAINPU_SegmentForeground,
    "BSAINPU_PoseDetect": BSAINPU_PoseDetect,
    "BSAINPU_DepthEstimate": BSAINPU_DepthEstimate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAINPU_LLMGenerate": "BSAI NPU LLM 决策文本",
    "BSAINPU_SegmentForeground": "BSAI NPU 前景抠图 (RMBG)",
    "BSAINPU_PoseDetect": "BSAI NPU 人体姿态 (YOLO11)",
    "BSAINPU_DepthEstimate": "BSAI NPU 深度估计 (DepthAnything)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# 插件加载完成后再补一次路由注册（此时 PromptServer.instance 已就绪）
_register_routes()
