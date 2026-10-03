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


# ---------------- 注册路由（容错：极端情况下 instance 不可用仍可加载） ----------------
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
        ]:
            try:
                routes.add_route(method, path, handler)
            except Exception:
                # 路径已被占用（如 ComfyUI 自带 /health）：尝试覆盖追加，仍失败则跳过
                try:
                    res = routes.get(path)
                    if res is not None:
                        res.add_route(method, path, handler)
                except Exception:
                    pass
        print("[BSAI-NPU-Service] NPU 服务已挂载到 ComfyUI: GET /health POST /detect_face /landmark_106 /recog_r50 /gender_age (device=%s)" % _svc.device)
    else:
        print("[BSAI-NPU-Service] PromptServer.instance 未就绪，路由延迟到节点触发注册")
except Exception as e:
    print("[BSAI-NPU-Service] 路由注册容错跳过:", e)
