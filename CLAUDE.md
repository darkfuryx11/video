# Video-to-Prompt

## จุดประสงค์
วิเคราะห์วีดีโอด้วย AI Vision Model สร้าง prompt / คำอธิบาย / tags อัตโนมัติ
Fork/extend จาก nanofatdog/video-to-prompt

## Stack
- Python 3.10+
- Gradio (Web UI — port 7860)
- llama.cpp (local inference)
- Qwen-VL (Vision Language Model)
- FastAPI (REST API)
- FFmpeg (video processing)

## ไฟล์หลัก
| ไฟล์ | หน้าที่ |
|---|---|
| `video_to_prompt_webui.py` | Gradio Web UI |
| `video_to_prompt.py` | Core logic |
| `api.py` | FastAPI REST endpoints |
| `install.sh` | Auto-install script |
| `requirements.txt` | Python dependencies |

## Run
```bash
# Web UI
python video_to_prompt_webui.py

# REST API
python api.py

# CLI
python video_to_prompt.py --video <path>
```

## Prerequisites
- llama.cpp server running (vision model เช่น Qwen-VL)
- FFmpeg ติดตั้งแล้ว
- Python 3.10+

## Deploy
- Local only (llama.cpp required)

## Repo
- Origin: https://github.com/nanofatdog/video-to-prompt
- Fork: darkfuryx11/video (TODO)

## หมายเหตุ
- ต้องรัน llama.cpp server แยกก่อนใช้งาน
