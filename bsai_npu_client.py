# -*- coding: utf-8 -*-
"""
BSAI NPU Client —— 共享 NPU 推理客户端
========================================
任何 BSAI 插件只需 import 本文件即可使用 NPU 人脸能力：

    from bsai_npu_client import npu, npu_available

    if npu_available():
        faces = npu.detect_faces(image_bgr)
        for f in faces:
            pts = npu.landmark_106(image_bgr, f["bbox"])
            gender_age = npu.gender_age(image_bgr, f["bbox"])

两种模式自动切换：
1) 同进程直连（BSAI-NPU-Service 已加载为 ComfyUI 插件）→ 零 HTTP 开销
2) HTTP 回退（NPU 服务独立运行在 8191 端口）
3) 服务未启动时自动拉起子进程

模型懒加载，首次调用自动编译到 NPU。
"""

import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

import numpy as np


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
NPU_PORT = 8191
NPU_HOST = "127.0.0.1"

# 单例
_svc = None
_direct = None  # True=同进程直连, False=HTTP, None=未探测


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _comfy_root():
    """ComfyUI 上级根目录（python_embeded 所在目录）"""
    here = os.path.dirname(os.path.abspath(__file__))
    # here = ComfyUI/ 或 ComfyUI/custom_nodes/BSAI-NPU-Service/
    # 逐级往上找 python_embeded
    d = here
    for _ in range(4):
        if os.path.exists(os.path.join(d, "python_embeded", "python.exe")) or \
           os.path.exists(os.path.join(d, "python_embeded_cuda", "python.exe")):
            return d
        d = os.path.dirname(d)
    return here


def _npu_service_dir():
    """定位 BSAI-NPU-Service 目录"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "BSAI-NPU-Service"),                      # 在 custom_nodes/ 里
        os.path.join(here, "custom_nodes", "BSAI-NPU-Service"),     # 在 ComfyUI/ 根
        os.path.join(here, "..", "custom_nodes", "BSAI-NPU-Service"), # 在 custom_nodes/BSAI-XXX/ 里
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "npu_service.py")):
            return os.path.abspath(c)
    return candidates[0]


def _npu_python():
    """找到 ComfyUI 嵌入式 Python"""
    root = _comfy_root()
    for p in [
        os.path.join(root, "python_embeded", "python.exe"),
        os.path.join(root, "python_embeded_cuda", "python.exe"),
    ]:
        if os.path.exists(p):
            return p
    return sys.executable


def _try_direct():
    """尝试同进程直连（BSAI-NPU-Service 已作为 ComfyUI 插件加载）"""
    global _direct, _svc
    if _direct is not None:
        return _direct

    # 直接从已加载的插件模块取单例
    try:
        import importlib
        mod = importlib.import_module("custom_nodes.BSAI-NPU-Service")
        if hasattr(mod, "_svc"):
            _svc = mod._svc
            _direct = True
            return True
    except Exception:
        pass

    # 尝试直接 import npu_service 模块
    try:
        sys.path.insert(0, _npu_service_dir())
        from npu_service import NPUService
        _svc = NPUService()
        _direct = True
        return True
    except Exception:
        pass

    _direct = False
    return False


def _http_post(path, payload):
    url = "http://%s:%d%s" % (NPU_HOST, NPU_PORT, path)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _http_get(path):
    url = "http://%s:%d%s" % (NPU_HOST, NPU_PORT, path)
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def _ensure_service():
    """确保 NPU 服务在线；不在线则自动拉起"""
    if _try_direct():
        return True
    try:
        _http_get("/health")
        return True
    except Exception:
        pass

    # 自动拉起
    npu_script = os.path.join(_npu_service_dir(), "npu_service.py")
    if os.path.exists(npu_script):
        try:
            py = _npu_python()
            subprocess.Popen(
                [py, npu_script, "--port", str(NPU_PORT)],
                creationflags=0x08000000 | 0x00000200,  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
            )
            # 等待就绪（最多 10 秒）
            for _ in range(20):
                time.sleep(0.5)
                try:
                    _http_get("/health")
                    return True
                except Exception:
                    continue
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------
def npu_available():
    """NPU 服务是否可用"""
    return _ensure_service()


def npu_status():
    """返回服务状态 dict"""
    if not _ensure_service():
        return {"status": "offline"}
    if _direct and _svc is not None:
        return _svc.status()
    try:
        return _http_get("/health")
    except Exception:
        return {"status": "offline"}


class _NPUClient:
    """统一客户端接口（同进程/HTTP 自动切换）"""

    # ---- 人脸检测 ----
    def detect_faces(self, image_bgr):
        """输入 BGR HxWx3 uint8，返回 [{score, bbox:[x1,y1,x2,y2], kps:[[x,y]*5}]"""
        if not _ensure_service():
            return []
        if _direct and _svc is not None:
            return _svc.detect_faces(image_bgr)
        b64 = _encode_image_b64(image_bgr)
        return _http_post("/detect_face", {"image": b64}).get("faces", [])

    # ---- 106 点关键点 ----
    def landmark_106(self, image_bgr, bbox):
        """输入 BGR 图 + bbox [x1,y1,x2,y2]，返回 [[x,y], ...] ×106"""
        if not _ensure_service():
            return []
        if _direct and _svc is not None:
            return _svc.landmark_106(image_bgr, bbox)
        b64 = _encode_image_b64(image_bgr)
        r = _http_post("/landmark_106", {"image": b64, "bbox": list(bbox)})
        return r.get("points", [])

    # ---- 人脸识别 ----
    def recognize(self, image_bgr, bbox):
        """输入 BGR 图 + bbox，返回 512 维归一化特征向量 list[float]"""
        if not _ensure_service():
            return []
        if _direct and _svc is not None:
            return _svc.recog_r50(image_bgr, bbox)
        b64 = _encode_image_b64(image_bgr)
        r = _http_post("/recog_r50", {"image": b64, "bbox": list(bbox)})
        return r.get("embedding", [])

    # ---- 性别年龄 ----
    def gender_age(self, image_bgr, bbox):
        """输入 BGR 图 + bbox，返回 {gender: male/female, age: int}"""
        if not _ensure_service():
            return {}
        if _direct and _svc is not None:
            return _svc.gender_age(image_bgr, bbox)
        b64 = _encode_image_b64(image_bgr)
        return _http_post("/gender_age", {"image": b64, "bbox": list(bbox)})

    # ---- YOLO11n-pose 人体姿态（NPU） ----
    def detect_pose(self, image_bgr):
        """输入 BGR 图，返回 [{score, bbox, keypoints:[[x,y,c]*17], person}]"""
        if not _ensure_service():
            return []
        if _direct and _svc is not None:
            return _svc.detect_pose(image_bgr)
        b64 = _encode_image_b64(image_bgr)
        return _http_post("/detect_pose", {"image": b64}).get("poses", [])

    # ---- NPU LLM 决策文本生成（Qwen3-4B 对称 INT4） ----
    def llm_generate(self, prompt, max_new_tokens=128, temperature=0.7, top_p=0.9,
                     enable_thinking=False, seed=None):
        """NPU 文本生成，返回 {text, tokens, secs, device, model}"""
        if not _ensure_service():
            return {"text": "", "error": "NPU service offline"}
        if _direct and _svc is not None:
            return _svc.llm_generate(prompt, max_new_tokens=max_new_tokens,
                                     temperature=temperature, top_p=top_p,
                                     enable_thinking=enable_thinking, seed=seed)
        r = _http_post("/llm_generate", {
            "prompt": prompt, "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature), "top_p": float(top_p),
            "enable_thinking": bool(enable_thinking), "seed": seed,
        })
        return r.get("result", {"text": r.get("error", "")})

    # ---- RMBG-1.4 前景抠图（NPU） ----
    def segment_foreground(self, image_bgr):
        """输入 BGR 图，返回 (前景 BGR 图, alpha HxW float32 0-1)"""
        if not _ensure_service():
            h, w = image_bgr.shape[:2]
            return image_bgr, np.zeros((h, w), dtype=np.float32)
        if _direct and _svc is not None:
            return _svc.segment_foreground(image_bgr)
        b64 = _encode_image_b64(image_bgr)
        r = _http_post("/segment_foreground", {"image": b64})
        fg_b64 = r.get("fg", "")
        alpha_b64 = r.get("alpha", "")
        fg = _decode_image_bgr_b64(fg_b64)
        try:
            alpha = np.frombuffer(base64.b64decode(alpha_b64), np.uint8)
            import cv2 as _cv2
            alpha = _cv2.imdecode(alpha, _cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        except Exception:
            alpha = np.zeros((fg.shape[0], fg.shape[1]), dtype=np.float32)
        return fg, alpha

    # ---- Depth-Anything-V2-Small 深度估计（NPU） ----
    def detect_depth(self, image_bgr):
        """输入 BGR 图，返回 (depth 灰度图 HxW uint8 0-255, meta dict)"""
        if not _ensure_service():
            h, w = image_bgr.shape[:2]
            return np.zeros((h, w), dtype=np.uint8), {"error": "NPU service offline"}
        if _direct and _svc is not None:
            return _svc.detect_depth(image_bgr)
        b64 = _encode_image_b64(image_bgr)
        r = _http_post("/detect_depth", {"image": b64})
        depth_b64 = r.get("depth", "")
        gray = _decode_image_bgr_b64(depth_b64)
        if gray.ndim == 3:
            gray = gray[:, :, 0]
        return gray, r.get("meta", {})

    # ---- 便捷：检测+关键点+性别年龄一条龙 ----
    def analyze(self, image_bgr):
        """一次性检测人脸并返回完整信息列表：
        [{score, bbox, kps, landmarks_106, gender, age, embedding}]
        """
        results = []
        for f in self.detect_faces(image_bgr):
            bbox = f["bbox"]
            entry = {
                "score": f["score"],
                "bbox": bbox,
                "kps": f["kps"],
                "landmarks_106": self.landmark_106(image_bgr, bbox),
            }
            try:
                ga = self.gender_age(image_bgr, bbox)
                entry["gender"] = ga.get("gender", "unknown")
                entry["age"] = ga.get("age", 0)
            except Exception:
                entry["gender"] = "unknown"
                entry["age"] = 0
            results.append(entry)
        return results


# 全局单例
npu = _NPUClient()


# ---------------------------------------------------------------------------
# 图像编码（HTTP 模式用）
# ---------------------------------------------------------------------------
def _encode_image_b64(image_bgr):
    """BGR numpy -> base64 jpeg"""
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    except ImportError:
        pass
    # PIL fallback
    from PIL import Image
    import io
    pil = Image.fromarray(image_bgr[:, :, ::-1])
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _decode_image_bgr_b64(data):
    """base64 jpeg -> BGR numpy"""
    try:
        import cv2 as _cv2
        raw = np.frombuffer(base64.b64decode(data), np.uint8)
        img = _cv2.imdecode(raw, _cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    from PIL import Image
    import io
    pil = Image.open(io.BytesIO(base64.b64decode(data)))
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return np.asarray(pil)[:, :, ::-1]


# ---------------------------------------------------------------------------
# 独立运行测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("BSAI NPU Client self-test")
    print("Status:", npu_status())
    print("Available:", npu_available())
