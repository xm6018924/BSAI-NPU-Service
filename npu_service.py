# -*- coding: utf-8 -*-
"""
BSAI NPU Service —— Intel AI Boost 人脸服务
=================================================
- 人脸检测 face-detect-10g（SCRFD 结构，OpenVINO IR，NPU 设备）
- 人脸关键点 face-landmark-106
- 人脸识别 face-recog-r50
- 性别年龄 gender-age

两种运行形态：
1) 作为 ComfyUI 插件（BSAI-NPU-Service/__init__.py 导入本模块，
   把 /health、/detect_face 等路由注册到 ComfyUI 的 8191 端口 —— FaceRefine 直接命中）
2) 独立服务（python npu_service.py [--port 8192]），供手动/调试
"""

import argparse
import base64
import json
import math
import os
import threading
import time
from collections import namedtuple

import numpy as np

try:
    from openvino import Core
except Exception:
    Core = None

MODELS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "NPU")
LLM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "LLM")

# 可用模型清单（目录名 -> 说明），/health 的 npu_models 列表来源
NPU_MODELS = [
    "face-detect-10g",
    "face-detect",
    "face-landmark-106",
    "face-3d-68",
    "face-recog-r50",
    "gender-age",
    "rmbg-1.4",
    "yolo11-pose",
    "depthanything-v2-small",
    "denoise-x1",
    "enhance-x1",
]

# NPU 兼容对称 INT4 的 Qwen3-4B（optimum --sym --group-size -1 --ratio 1.0 导出）
LLM_NPU_IR = os.path.join(LLM_ROOT, "Qwen3-4B-NPU-SYM-IR")

try:
    import openvino_genai as _ov_genai
except Exception:
    _ov_genai = None

DETECT_INPUT = 640   # SCRFD 输入边长
DETECT_STRIDES = (8, 16, 32)
DETECT_NUM_ANCHORS = 2   # 每个位置 anchor 数（12800 = 80*80*2）
# 官方 SCRFD 默认 det_thresh=0.5；FaceRefine 场景为视频人脸（多为人脸特写），
# 取 0.35 兼顾小脸/模糊帧，避免漏检
DETECT_SCORE_THRESH = 0.35
DETECT_NMS_THRESH = 0.45
DETECT_NMS_KEEP = 100


def _core():
    if Core is None:
        raise RuntimeError("openvino 未安装: pip install openvino")
    return Core()


def _available_devices(core):
    try:
        return core.available_devices
    except Exception:
        return []


def _has_device(core, name):
    return name in _available_devices(core)


class NPUService:
    """NPU 推理服务（懒加载模型，线程安全：同一 compiled model 并发推理由 OpenVINO 保证）

    设备降级链：NPU → GPU（Intel 核显/XPU）→ CPU
    - NPU 最快最低功耗
    - GPU（核显）比 CPU 快 5-10 倍，OpenVINO 对 SCRFD/R50 算子支持完整
    - CPU 兜底，保证服务不中断
    """

    # 优先级：首选 NPU，其次 GPU（核显），最后 CPU
    DEVICE_PRIORITY = ["NPU", "GPU", "CPU"]

    def __init__(self, device="NPU", models_root=MODELS_ROOT):
        self.requested_device = device
        self.device = device
        self.models_root = models_root
        self._core = None
        self._compiled = {}
        self._lazy_lock = None  # 由 aiohttp 事件循环外调用，无并发竞争；保留字段兼容
        self._inflight = 0        # 当前在途推理数（队列深度）
        self._total_calls = 0     # 累计推理调用数
        self._inflight_lock = threading.Lock()
        self._llm_pipe = None     # openvino_genai.LLMPipeline 单例（NPU LLM）
        self._llm_lock = threading.Lock()
        self._llm_total_calls = 0

    # ---------------- 基础设施 ----------------
    @property
    def core(self):
        if self._core is None:
            c = _core()
            # 按优先级选择第一个可用设备：NPU → GPU → CPU
            avail = _available_devices(c)
            chosen = "CPU"  # CPU 永远可用
            for d in self.DEVICE_PRIORITY:
                if d == "CPU":
                    break
                if d in avail:
                    chosen = d
                    break
            # 如果用户显式指定了 GPU 或 CPU（命令行 --device），尊重用户选择
            if self.requested_device in ("GPU", "CPU") and self.requested_device in avail:
                chosen = self.requested_device
            self.device = chosen
            print("[BSAI-NPU-Service] device selected: %s (available: %s)" % (chosen, avail))
            # 持久化 OpenVINO 编译缓存：首次编译后写入 models/NPU/.cache，
            # 后续启动跳过 NPU 重编译，缩短预热到秒级
            try:
                cache_dir = os.path.join(self.models_root, ".cache")
                os.makedirs(cache_dir, exist_ok=True)
                c.set_property(self.device, "CACHE_DIR", cache_dir)
            except Exception:
                pass
            self._core = c
        return self._core

    def model_path(self, name):
        return os.path.join(self.models_root, name, "openvino_model.xml")

    def get_compiled(self, name, device=None):
        """懒加载 + 编译缓存。
        每模型按降级链尝试：首选 self.device，编译失败自动退到 GPU → CPU。
        """
        xml = self.model_path(name)
        if not os.path.exists(xml):
            raise FileNotFoundError("NPU 模型缺失: %s" % xml)
        # 尝试设备列表：显式指定 > 主设备 > GPU > CPU
        candidates = []
        if device:
            candidates.append(device)
        candidates.append(self.device)
        for d in self.DEVICE_PRIORITY:
            if d not in candidates:
                candidates.append(d)
        last_err = None
        for dev in candidates:
            key = (name, dev)
            if key in self._compiled:
                return self._compiled[key]
            try:
                self._compiled[key] = self.core.compile_model(xml, dev)
                print("[BSAI-NPU-Service] model %s compiled on %s" % (name, dev))
                return self._compiled[key]
            except Exception as e:
                last_err = e
                print("[BSAI-NPU-Service] model %s failed on %s: %s" % (name, dev, e))
                continue
        raise RuntimeError("model %s failed on all devices: %s" % (name, last_err))

    def status(self):
        dev = self.device
        if self._core is not None:
            try:
                dev = self.core.get_property(self.device, "FULL_DEVICE_NAME")
            except Exception:
                pass
        with self._inflight_lock:
            inflight = self._inflight
            total = self._total_calls
        # ready = 核心人脸检测模型已完成编译（首次推理后为 True）
        ready = ("face-detect-10g", self.device) in self._compiled
        return {
            "status": "online" if self._core is not None else "degraded",
            "ready": bool(ready),
            "core": "ok" if Core is not None else "missing",
            "device": self.device,
            "device_name": dev,
            "npu_models": [m for m in NPU_MODELS if os.path.isdir(os.path.join(self.models_root, m))],
            "inflight": inflight,
            "queue_depth": inflight,
            "total_calls": total,
            "ts": time.time(),
        }

    def ready(self):
        """就绪探针：推理通道可用 + 核心检测模型文件在位（可立即分发）"""
        st = self.status()
        try:
            self.core  # 确保推理通道可建（Core 可用 / 设备可枚举）
            xml = self.model_path("face-detect-10g")
            ok = os.path.exists(xml)
        except Exception:
            ok = False
        return {
            "ready": bool(ok),
            "status": st["status"],
            "device": st["device"],
            "compiled_models": list(self._compiled.keys()),
            "ts": time.time(),
        }

    # ---------------- 人脸检测（SCRFD） ----------------
    @staticmethod
    def _generate_anchors(stride, fmap, num_anchors=2):
        """标准 insightface SCRFD anchor：每个格子 (cx,cy)，num_anchors 份"""
        out = []
        for i in range(fmap):
            for j in range(fmap):
                cx = (j + 0.5) * stride
                cy = (i + 0.5) * stride
                for _ in range(num_anchors):
                    out.append([cx, cy])
        return np.array(out, dtype=np.float32)

    @staticmethod
    def _distance2bbox(points, distance):
        """SCRFD bbox 解码：distance = [left, top, right, bottom]"""
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _distance2kps(points, distance):
        """SCRFD 关键点解码：distance = [5 点 × (dx,dy)]"""
        preds = []
        for idx in range(5):
            preds.append(points[:, 0] + distance[:, idx * 2])
            preds.append(points[:, 1] + distance[:, idx * 2 + 1])
        return np.stack(preds, axis=-1).reshape(-1, 5, 2)

    @staticmethod
    def _nms(boxes, scores, thresh):
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[np.where(iou <= thresh)[0] + 1]
        return keep

    def detect_faces(self, img_bgr):
        """输入 BGR HxWx3 uint8（640 内任意尺寸，内部缩放），返回检测结果列表"""
        with self._inflight_lock:
            self._inflight += 1
            self._total_calls += 1
        try:
            return self._detect_faces_impl(img_bgr)
        finally:
            with self._inflight_lock:
                self._inflight -= 1

    def _detect_faces_impl(self, img_bgr):
        compiled = self.get_compiled("face-detect-10g")
        h0, w0 = img_bgr.shape[:2]
        scale = DETECT_INPUT / max(h0, w0)
        nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
        if nh != h0 or nw != w0:
            # 最近邻/双线性缩放（保持宽高比，居中填充）
            img = np.zeros((DETECT_INPUT, DETECT_INPUT, 3), dtype=np.uint8)
            img[:nh, :nw] = np.asarray(ImageResize(img_bgr, nh, nw))
        else:
            img = img_bgr.astype(np.uint8)
        # insightface SCRFD 预处理（cv2.dnn.blobFromImage 语义，实测公式 (x-mean)*scale）：
        # blobFromImage(img, 1/128, size, (127.5,127.5,127.5), swapRB=True) => (x-127.5)/128, RGB
        rgb = img[:, :, ::-1].astype(np.float32)
        blob = (rgb - 127.5) * (1.0 / 128.0)
        blob = np.transpose(blob, (2, 0, 1))[None]  # 1,3,640,640

        outs = compiled([blob])
        # 输出按类型分组：先 3 个 score(8/16/32)、再 3 个 bbox、再 3 个 kps；
        # 不依赖顺序/名字，按输出 shape 动态分类（[12800|3200|800, 1|4|10]）
        det = {}
        for o in compiled.outputs:
            arr = np.asarray(outs[o.any_name])
            sh = arr.shape
            if len(sh) != 2:
                continue
            n, d = sh
            if n not in (12800, 3200, 800) or d not in (1, 4, 10):
                continue
            stride = {12800: 8, 3200: 16, 800: 32}[n]
            kind = {1: "score", 4: "bbox", 10: "kps"}[d]
            det[(stride, kind)] = arr.reshape(n, d)

        faces = []
        for stride in DETECT_STRIDES:
            if (stride, "score") not in det:
                continue
            score = det[(stride, "score")].reshape(-1)
            bbox = det[(stride, "bbox")]
            kps = det[(stride, "kps")]
            fmap = DETECT_INPUT // stride
            num_per_pos = bbox.shape[0] // (fmap * fmap)
            anchors = self._generate_anchors(stride, fmap, num_per_pos)
            keep = np.where(score > DETECT_SCORE_THRESH)[0]
            if len(keep) == 0:
                continue
            anchors = anchors[keep]
            # 官方 SCRFD：bbox/kps 预测值需先乘 stride 再按 distance 解码
            boxes = self._distance2bbox(anchors, bbox[keep] * stride)
            kpss = self._distance2kps(anchors, kps[keep] * stride)
            sc = score[keep]
            # NMS
            idx = self._nms(boxes, sc, DETECT_NMS_THRESH)
            for i in idx[:DETECT_NMS_KEEP]:
                x1, y1, x2, y2 = boxes[i]
                # 还原到原图坐标
                x1 = x1 / scale if scale else x1
                y1 = y1 / scale if scale else y1
                x2 = x2 / scale if scale else x2
                y2 = y2 / scale if scale else y2
                k = kpss[i] / scale if scale else kpss[i]
                faces.append({
                    "score": float(sc[i]),
                    "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
                    "kps": [[round(float(v), 1) for v in p] for p in k.tolist()],
                })
        # 全尺度合并后按分数排序
        faces.sort(key=lambda f: f["score"], reverse=True)
        return faces

    # ---------------- 其他模型（关键点 / 识别 / 性别年龄） ----------------
    def _crop_blob(self, model, img_bgr, box, pad_frac=0.0):
        """裁剪 bbox 并按模型输入 shape 缩放，返回 (blob, crop_region)"""
        x1, y1, x2, y2 = [int(v) for v in box]
        if pad_frac:
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            pad = int(pad_frac * max(w, h))
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(img_bgr.shape[1], x2 + pad); y2 = min(img_bgr.shape[0], y2 + pad)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None, (x1, y1, x2, y2)
        compiled = self.get_compiled(model)
        # 动态读模型输入 HxW（兼容不同模型：192x192 / 112x112 / 96x96 等）
        try:
            sh = list(compiled.inputs[0].get_partial_shape().to_shape())
            ih, iw = int(sh[2]), int(sh[3])
        except Exception:
            ih = iw = 112
        resized = np.asarray(ImageResize(crop, ih, iw), dtype=np.float32)
        rgb = resized[:, :, ::-1]
        blob = (rgb - 127.5) * (1.0 / 128.0)
        blob = np.transpose(blob, (2, 0, 1))[None]
        return blob, (x1, y1, x2, y2, ih, iw)

    def landmark_106(self, img_bgr, box):
        """人脸关键点（106 点）：需对齐裁剪；box 为原图 bbox"""
        blob, reg = self._crop_blob("face-landmark-106", img_bgr, box, pad_frac=0.2)
        if blob is None:
            return []
        x1, y1, x2, y2, ih, iw = reg
        out = self.get_compiled("face-landmark-106")([blob])
        pts = np.asarray(out[0]).reshape(-1, 2)
        scale_x = (x2 - x1) / float(iw)
        scale_y = (y2 - y1) / float(ih)
        return [[round(x1 + p[0] * scale_x, 1), round(y1 + p[1] * scale_y, 1)] for p in pts]

    def recog_r50(self, img_bgr, box):
        """人脸识别向量（512d），先裁剪对齐（简化：直接裁剪缩放）"""
        blob, reg = self._crop_blob("face-recog-r50", img_bgr, box)
        if blob is None:
            return []
        out = self.get_compiled("face-recog-r50")([blob])
        emb = np.asarray(out[0]).reshape(-1)
        norm = float(np.linalg.norm(emb) + 1e-9)
        return [round(float(v) / norm, 6) for v in emb]

    def gender_age(self, img_bgr, box):
        """性别年龄（简化：按 bbox 裁剪缩放）"""
        blob, reg = self._crop_blob("gender-age", img_bgr, box)
        if blob is None:
            return {}
        out = self.get_compiled("gender-age")([blob])
        vals = np.asarray(out[0]).reshape(-1)
        # 官方 insightface genderage 解码：gender = M if out[0]>0；age = int(out[1]*100)
        gender = "female" if vals[0] < 0 else "male"
        return {"gender": gender, "age": int(abs(vals[1]) * 100)}

    # ---------------- NPU LLM（Qwen3-4B 对称 INT4，决策文本生成） ----------------
    def get_llm_pipe(self):
        """懒加载 openvino_genai LLMPipeline（NPU 编译一次约 40s，缓存单例）"""
        if self._llm_pipe is not None:
            return self._llm_pipe
        with self._llm_lock:
            if self._llm_pipe is not None:
                return self._llm_pipe
            if _ov_genai is None:
                raise RuntimeError("openvino-genai 未安装: pip install openvino-genai openvino-tokenizers")
            xml = os.path.join(LLM_NPU_IR, "openvino_model.xml")
            if not os.path.exists(xml):
                raise FileNotFoundError("NPU LLM IR 缺失: %s（需 optimum 导出 --sym --group-size -1 --ratio 1.0）" % xml)
            t0 = time.time()
            self._llm_pipe = _ov_genai.LLMPipeline(LLM_NPU_IR, self.device)
            print("[BSAI-NPU-Service] LLM %s compiled on %s %.1fs" %
                  (os.path.basename(LLM_NPU_IR), self.device, time.time() - t0))
            return self._llm_pipe

    def llm_generate(self, prompt, max_new_tokens=128, temperature=0.7, top_p=0.9,
                     enable_thinking=False, seed=None):
        """NPU 文本生成。返回 {text, tokens, secs, device}。
        enable_thinking=False（默认）：system 指令压制思考 + 剥离思考残片，直接给答案；
        enable_thinking=True：允许模型自由思考。"""
        import re
        t0 = time.time()
        pipe = self.get_llm_pipe()
        cfg = _ov_genai.GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
        )
        if seed is not None:
            cfg.rng_seed = int(seed)
        with self._llm_lock:
            self._llm_total_calls += 1
            try:
                if bool(enable_thinking):
                    text = pipe.generate(prompt, cfg)
                else:
                    hist = _ov_genai.ChatHistory([
                        {"role": "system",
                         "content": "You must answer directly. Do NOT output any thinking process or reasoning. "
                                    "Give the answer immediately, without <think> blocks."},
                        {"role": "user", "content": prompt},
                    ])
                    text = pipe.generate(hist, cfg)
            except Exception:
                text = pipe.generate(prompt, cfg)
        text = str(text)
        if not bool(enable_thinking):
            # 剥离思考块（含模板注入的空块与模型自写残片）
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
            text = text.replace("</think>", "").replace("<think>", "")
        text = text.strip()
        return {
            "text": text,
            "tokens": int(max_new_tokens),
            "secs": round(time.time() - t0, 3),
            "device": self.device,
            "model": os.path.basename(LLM_NPU_IR),
        }

    # ---------------- YOLO11n-pose 人体姿态（NPU） ----------------
    YOLO_INPUT = 640
    YOLO_NUM_KPTS = 17
    YOLO_SCORE_THRESH = 0.25
    YOLO_NMS_THRESH = 0.45

    def detect_pose(self, img_bgr):
        """YOLO11n-pose 人体检测+17 关键点。输入 BGR HxWx3 uint8 ->
        [{score, bbox:[x1,y1,x2,y2], keypoints:[[x,y,c]*17], person:bool}]"""
        with self._inflight_lock:
            self._inflight += 1
            self._total_calls += 1
        try:
            compiled = self.get_compiled("yolo11-pose")
            h0, w0 = img_bgr.shape[:2]
            # letterbox 640（灰边 114，居中放置）
            scale = min(self.YOLO_INPUT / h0, self.YOLO_INPUT / w0)
            nh, nw = max(1, int(round(h0 * scale))), max(1, int(round(w0 * scale)))
            y_off = (self.YOLO_INPUT - nh) // 2
            x_off = (self.YOLO_INPUT - nw) // 2
            canvas = np.full((self.YOLO_INPUT, self.YOLO_INPUT, 3), 114, dtype=np.uint8)
            canvas[y_off:y_off + nh, x_off:x_off + nw] = np.asarray(ImageResize(img_bgr, nh, nw))
            rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
            blob = np.transpose(rgb, (2, 0, 1))[None]

            outs = compiled([blob])
            arr = np.asarray(outs[0])
            if arr.ndim == 3:
                arr = arr[0]
            # YOLO11 输出 (56 特征, 8400 anchor)：特征在行、anchor 在列 -> 转置为 (anchor, 特征)
            if arr.shape[0] == 56:
                pred = arr.T
            else:
                pred = arr
            # pred: (N, 56) -> box(4) + cls(1, person, 已 sigmoid) + kpt_x(17) + kpt_y(17) + kpt_c(17)
            boxes_xywh = pred[:, 0:4]
            conf = pred[:, 4]  # person 置信度（ONNX 输出已过 sigmoid，0-1）
            kpt_x = pred[:, 5:5 + self.YOLO_NUM_KPTS]
            kpt_y = pred[:, 5 + self.YOLO_NUM_KPTS:5 + 2 * self.YOLO_NUM_KPTS]
            kpt_c = pred[:, 5 + 2 * self.YOLO_NUM_KPTS:5 + 3 * self.YOLO_NUM_KPTS]

            keep = np.where(conf > self.YOLO_SCORE_THRESH)[0]
            results = []
            if len(keep):
                bx = boxes_xywh[keep]
                x1 = bx[:, 0] - bx[:, 2] / 2.0
                y1 = bx[:, 1] - bx[:, 3] / 2.0
                x2 = bx[:, 0] + bx[:, 2] / 2.0
                y2 = bx[:, 1] + bx[:, 3] / 2.0
                box = np.stack([x1, y1, x2, y2], axis=-1)
                sc = conf[keep]
                idx = self._nms(box, sc, self.YOLO_NMS_THRESH)
                for i in idx:
                    x1, y1, x2, y2 = box[i]
                    # 640 坐标还原到原图（letterbox 逆变换）
                    x1 = (x1 - x_off) / scale
                    y1 = (y1 - y_off) / scale
                    x2 = (x2 - x_off) / scale
                    y2 = (y2 - y_off) / scale
                    kpts = []
                    for j in range(self.YOLO_NUM_KPTS):
                        px = (kpt_x[keep][i, j] - x_off) / scale
                        py = (kpt_y[keep][i, j] - y_off) / scale
                        pc = float(kpt_c[keep][i, j])  # 可见性（已 sigmoid）
                        kpts.append([round(float(px), 1), round(float(py), 1), round(float(pc), 4)])
                    results.append({
                        "score": round(float(sc[i]), 4),
                        "bbox": [round(float(x1), 1), round(float(y1), 1),
                                 round(float(x2), 1), round(float(y2), 1)],
                        "keypoints": kpts,
                        "person": bool(float(sc[i]) > 0.3),
                    })
            results.sort(key=lambda r: r["score"], reverse=True)
            return results
        finally:
            with self._inflight_lock:
                self._inflight -= 1

    # ---------------- RMBG-1.4 抠图（NPU） ----------------
    RMBG_INPUT = 1024
    RMBG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    RMBG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def segment_foreground(self, img_bgr):
        """RMBG-1.4 前景分割：输入 BGR HxWx3 uint8 → (前景 BGR 图, alpha HxW float32 0-1)"""
        with self._inflight_lock:
            self._inflight += 1
            self._total_calls += 1
        try:
            compiled = self.get_compiled("rmbg-1.4")
            h0, w0 = img_bgr.shape[:2]
            # BiRefNet 前处理：resize 1024x1024（保持比例居中填充）+ RGB 0-1 + ImageNet 归一化
            img = np.zeros((self.RMBG_INPUT, self.RMBG_INPUT, 3), dtype=np.float32)
            scale = min(self.RMBG_INPUT / h0, self.RMBG_INPUT / w0)
            nh, nw = max(1, int(round(h0 * scale))), max(1, int(round(w0 * scale)))
            resized = np.asarray(ImageResize(img_bgr, nh, nw), dtype=np.float32) / 255.0
            y0 = (self.RMBG_INPUT - nh) // 2
            x0 = (self.RMBG_INPUT - nw) // 2
            img[y0:y0 + nh, x0:x0 + nw] = resized
            rgb = img[:, :, ::-1]
            blob = ((rgb - self.RMBG_MEAN) / self.RMBG_STD).transpose(2, 0, 1)[None].astype(np.float32)

            out = compiled([blob])
            raw = np.asarray(out[0]).reshape(self.RMBG_INPUT, self.RMBG_INPUT)
            # BiRefNet 输出已是 sigmoid 后的 0-1 前景概率（实测 p5≈0 / p95≈1）
            alpha = np.clip(raw, 0.0, 1.0).astype(np.float32)
            # 裁回原图比例 + 缩回原尺寸
            alpha = alpha[y0:y0 + nh, x0:x0 + nw]
            alpha = np.asarray(ImageResize((alpha * 255).astype(np.uint8), h0, w0), dtype=np.float32) / 255.0
            fg = img_bgr.astype(np.float32) * alpha[:, :, None]
            return np.clip(fg, 0, 255).astype(np.uint8), alpha
        finally:
            with self._inflight_lock:
                self._inflight -= 1

    # ---------------- Depth-Anything-V2-Small 深度估计（NPU） ----------------
    DEPTH_INPUT = 518
    DEPTH_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    DEPTH_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def detect_depth(self, img_bgr):
        """Depth-Anything-V2-Small 单目深度：输入 BGR HxWx3 uint8 ->
        (depth_map HxW uint8 0-255 灰度, meta{min,max,secs})"""
        with self._inflight_lock:
            self._inflight += 1
            self._total_calls += 1
        try:
            t0 = time.time()
            compiled = self.get_compiled("depthanything-v2-small")
            h0, w0 = img_bgr.shape[:2]
            # 前处理：等比缩放 518 + 黑边居中（与 HF DPTImageProcessor 对齐）
            scale = min(self.DEPTH_INPUT / h0, self.DEPTH_INPUT / w0)
            nh, nw = max(1, int(round(h0 * scale))), max(1, int(round(w0 * scale)))
            canvas = np.zeros((self.DEPTH_INPUT, self.DEPTH_INPUT, 3), dtype=np.float32)
            resized = np.asarray(ImageResize(img_bgr, nh, nw), dtype=np.float32) / 255.0
            y0 = (self.DEPTH_INPUT - nh) // 2
            x0 = (self.DEPTH_INPUT - nw) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = resized
            rgb = canvas[:, :, ::-1]
            blob = ((rgb - self.DEPTH_MEAN) / self.DEPTH_STD).transpose(2, 0, 1)[None].astype(np.float32)

            out = compiled([blob])
            depth = np.asarray(out[0]).reshape(self.DEPTH_INPUT, self.DEPTH_INPUT).astype(np.float32)
            # 裁回原比例
            depth = depth[y0:y0 + nh, x0:x0 + nw]
            dmin, dmax = float(depth.min()), float(depth.max())
            # min-max 归一化到 0-255 灰度（DepthAnything 输出为相对深度）
            span = (dmax - dmin) or 1.0
            gray = np.clip((depth - dmin) / span * 255.0, 0, 255).astype(np.uint8)
            # 缩回原图尺寸
            gray = np.asarray(ImageResize(gray, h0, w0))
            return gray, {"min": round(dmin, 4), "max": round(dmax, 4),
                          "secs": round(time.time() - t0, 3)}
        finally:
            with self._inflight_lock:
                self._inflight -= 1


# ---------------- 图像缩放（避免依赖 cv2：PIL/numpy 双保险） ----------------
def ImageResize(img, nh, nw):
    """缩放 BGR numpy 到 (nh, nw)，双线性（PIL 实现）"""
    try:
        from PIL import Image
        pil = Image.fromarray(img[:, :, ::-1])  # BGR -> RGB
        pil = pil.resize((nw, nh), Image.BILINEAR)
        return np.asarray(pil)[:, :, ::-1]  # RGB -> BGR
    except Exception:
        # 纯 numpy 最近邻（兜底）
        h, w = img.shape[:2]
        ys = (np.arange(nh) * h / nh).astype(int)
        xs = (np.arange(nw) * w / nw).astype(int)
        return img[ys][:, xs]


# ---------------- 独立服务入口 ----------------
def decode_image_bgr(data: bytes):
    """bytes -> BGR HxWx3 uint8（支持 b64 前缀）"""
    if isinstance(data, (str, bytes)) and (isinstance(data, str) and data.startswith("data:image") or
                                           isinstance(data, bytes) and data.startswith(b"data:image")):
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        data = data.split(",", 1)[1]
    if isinstance(data, str):
        data = base64.b64decode(data)
    img = np.frombuffer(data, np.uint8)
    try:
        import cv2
        bgr = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("image decode failed")
        return bgr
    except ImportError:
        from PIL import Image
        import io
        pil = Image.open(io.BytesIO(img))
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        return np.asarray(pil)[:, :, ::-1]


def main():
    ap.add_argument("--port", type=int, default=8192)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="NPU")
    args = ap.parse_args()

    svc = NPUService(device=args.device)

    async def health(_):
        return web.json_response(svc.status())

    async def health_ready(_):
        return web.json_response(svc.ready())

    async def detect_face(req):
        try:
            body = await req.json()
            img = decode_image_bgr(body.get("image", ""))
            faces = svc.detect_faces(img)
            return web.json_response({"count": len(faces), "faces": faces})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/health/ready", health_ready)
    app.router.add_post("/detect_face", detect_face)
    web.run_app(app, host=args.host, port=args.port)
    print("[BSAI-NPU-Service] listening on %s:%d device=%s" % (args.host, args.port, svc.device))


if __name__ == "__main__":
    from aiohttp import web
    main()
