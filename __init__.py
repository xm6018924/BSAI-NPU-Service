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

FaceRefine 会请求 http://127.0.0.1:8191/health 判断 NPU 服务是否在线；
本插件加载后该请求直接命中，FaceRefine 的自动拉起逻辑不会触发（无端口冲突）。
"""

import json
import os

import aiohttp
from aiohttp import web

from .npu_service import NPUService, decode_image_bgr

WEB_DIRECTORY = "./web" if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")) else None

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

_svc = NPUService()


# ---------------- 路由处理 ----------------
async def npu_health(_):
    return web.json_response(_svc.status())


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
