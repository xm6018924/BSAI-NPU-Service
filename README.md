# BSAI-NPU-Service

**English** | [中文](#中文)

---

## English

Intel AI Boost (NPU) face inference service for ComfyUI. Offloads face detection, landmark, recognition and gender/age tasks from GPU to the NPU, freeing VRAM for video generation.

### Features

- **Face Detection** — SCRFD-based 640×640 detector with NMS, tuned for video faces (threshold 0.35)
- **106-point Landmark** — Facial keypoint detection for face alignment
- **Face Recognition** — 512-dim R50 embedding vector (runs on CPU due to NPU operator issues)
- **Gender & Age** — Single-inference gender classification and age estimation
- **Zero GPU overhead** — All inference runs on Intel NPU via OpenVINO
- **Auto-fallback** — Gracefully degrades to CPU when NPU is unavailable
- **Dual mode** — ComfyUI plugin (mounts on :8191) or standalone HTTP service (:8192)

### Requirements

- Windows 10/11 with Intel NPU (Meteor Lake / Lunar Lake / Arrow Lake)
- Intel NPU driver installed
- ComfyUI embedded Python environment
- OpenVINO runtime

### Installation

1. Clone into your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xm6018924/BSAI-NPU-Service.git
```

2. Install dependencies:
```bash
# Using ComfyUI's embedded Python
python_embeded\python.exe -s -m pip install -r BSAI-NPU-Service\requirements.txt
```

3. Download OpenVINO IR models to `ComfyUI/models/NPU/`:
```
ComfyUI/models/NPU/
├── face-detect-10g/openvino_model.xml
├── face-landmark-106/openvino_model.xml
├── face-recog-r50/openvino_model.xml
└── gender-age/openvino_model.xml
```

4. Restart ComfyUI. On startup you should see:
```
[BSAI-NPU-Service] NPU 服务已挂载到 ComfyUI: GET /health POST /detect_face ...
```

### Usage

#### As ComfyUI plugin

The plugin automatically registers HTTP routes on ComfyUI's existing port (default 8191). No additional node appears in the workflow — it works behind the scenes for [BSAIFaceRefine](https://github.com/xm6018924) which calls these endpoints to detect faces on NPU.

Verify the service is online:
```bash
curl http://127.0.0.1:8191/health
```

Response:
```json
{
  "status": "online",
  "device": "NPU",
  "device_name": "Intel(R) AI Boost (NPU)",
  "npu_models": ["face-detect-10g", "face-landmark-106", "face-recog-r50", "gender-age"],
  "ts": 1727800000.0
}
```

#### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status and available models |
| POST | `/detect_face` | Face detection, returns bbox + 5-point keypoints |
| POST | `/landmark_106` | 106-point facial landmarks (requires bbox) |
| POST | `/recog_r50` | 512-dim face embedding (requires bbox) |
| POST | `/gender_age` | Gender and age estimate (requires bbox) |

**POST body example** (base64 image):
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "bbox": [120.5, 80.0, 350.2, 400.0]
}
```

#### Standalone mode

Run independently for debugging:
```bash
python npu_service.py --port 8192 --device NPU
```

### Notes

- `face-recog-r50` is forced to CPU because the NPU backend produces NaN outputs for this model (known OpenVINO operator issue).
- If no NPU is detected, the service automatically switches to CPU and reports this in `/health`.
- Models are lazily loaded on first request, so the first call to each endpoint has a small warmup delay.

---

## 中文

Intel AI Boost（NPU）人脸推理服务，作为 ComfyUI 插件运行。将人脸检测、关键点、识别和性别年龄推理从 GPU 卸载到 NPU，释放显存给视频生成。

### 功能特性

- **人脸检测** — 基于 SCRFD 的 640×640 检测器，带 NMS，针对视频人脸调优（阈值 0.35）
- **106 点关键点** — 人脸对齐关键点检测
- **人脸识别** — 512 维 R50 特征向量（因 NPU 算子问题强制走 CPU）
- **性别年龄** — 单次推理输出性别分类和年龄估计
- **零 GPU 开销** — 所有推理通过 OpenVINO 在 Intel NPU 上运行
- **自动降级** — 无 NPU 时自动切换 CPU，服务不中断
- **双模式** — ComfyUI 插件（挂载 :8191）或独立 HTTP 服务（:8192）

### 环境要求

- Windows 10/11，配备 Intel NPU（Meteor Lake / Lunar Lake / Arrow Lake）
- 已安装 Intel NPU 驱动
- ComfyUI 嵌入式 Python 环境
- OpenVINO 运行时

### 安装步骤

1. 克隆到 ComfyUI 的 `custom_nodes` 目录：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xm6018924/BSAI-NPU-Service.git
```

2. 安装依赖：
```bash
# 使用 ComfyUI 自带的 Python
python_embeded\python.exe -s -m pip install -r BSAI-NPU-Service\requirements.txt
```

3. 下载 OpenVINO IR 模型到 `ComfyUI/models/NPU/`：
```
ComfyUI/models/NPU/
├── face-detect-10g/openvino_model.xml
├── face-landmark-106/openvino_model.xml
├── face-recog-r50/openvino_model.xml
└── gender-age/openvino_model.xml
```

4. 重启 ComfyUI。启动日志中应出现：
```
[BSAI-NPU-Service] NPU 服务已挂载到 ComfyUI: GET /health POST /detect_face ...
```

### 使用方法

#### 作为 ComfyUI 插件

插件自动在 ComfyUI 现有端口（默认 8191）上注册 HTTP 路由。工作流中无需添加新节点——BSAIFaceRefine 会自动调用这些接口在 NPU 上检测人脸。

验证服务在线：
```bash
curl http://127.0.0.1:8191/health
```

返回示例：
```json
{
  "status": "online",
  "device": "NPU",
  "device_name": "Intel(R) AI Boost (NPU)",
  "npu_models": ["face-detect-10g", "face-landmark-106", "face-recog-r50", "gender-age"],
  "ts": 1727800000.0
}
```

#### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 服务状态和可用模型列表 |
| POST | `/detect_face` | 人脸检测，返回 bbox + 5 点关键点 |
| POST | `/landmark_106` | 106 点人脸关键点（需传入 bbox） |
| POST | `/recog_r50` | 512 维人脸特征向量（需传入 bbox） |
| POST | `/gender_age` | 性别和年龄估计（需传入 bbox） |

**POST 请求体示例**（base64 图片）：
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "bbox": [120.5, 80.0, 350.2, 400.0]
}
```

#### 独立模式

调试时可独立运行：
```bash
python npu_service.py --port 8192 --device NPU
```

### 注意事项

- `face-recog-r50` 因 NPU 后端算子输出 NaN 的已知问题，强制走 CPU 推理。
- 未检测到 NPU 时，服务自动切换 CPU，并在 `/health` 中如实上报实际设备。
- 模型采用懒加载策略，首次调用每个接口时有少量编译预热延迟。

---

## License

MIT
