# main.py
import time

from charset_normalizer.md import annotations

from voice_core.recorder_service import RecorderService
from voice_core.asr_client import ASRClient

# 初始化服务
recorder = RecorderService(max_duration=60)
asr_client = ASRClient()


# 可选：设置回调
def on_audio_ready(audio_bytes: bytes):
    """录音完成后自动调用 ASR"""
    print("🎤 录音完成，正在转录...")
    try:
        result = asr_client.transcribe_bytes(audio_bytes)
        if result.get("status_code") == 200:
            text = result["output"]["choices"][0]["message"]["content"][0]["text"]
            emotion = result["output"]["choices"][0]["message"]["annotations"][0]["emotion"]
            print(f"✅ 识别结果: {text}")
            print(f"<UNK> <UNK>: {emotion}")
        else:
            print(f"❌ ASR 失败: {result}")
    except Exception as e:
        print(f"❌ 转录异常: {e}")


recorder.on_stop = on_audio_ready


# ============ 模拟前端调用接口 ============

def start_recording():
    """🔘 前端调用：开始录音"""
    result = recorder.start()
    print(f"▶️  {result}")
    return result


def stop_recording():
    """⏹️ 前端调用：停止录音"""
    result = recorder.stop()
    print(f"⏹️  {result}")
    return result


def get_status():
    """📊 前端调用：查询状态"""
    return {"is_recording": recorder.is_recording()}


# ============ 测试入口 ============
if __name__ == "__main__":
    print("🎙️ 录音服务已就绪，输入命令控制：")
    print("  s      - 开始录音")
    print("  e      - 停止录音")
    print("  o      - 查看状态")
    print("  q      - 退出")

    while True:
        cmd = input("\n> ").strip().lower()
        if cmd == "s":
            start_recording()
        elif cmd == "e":
            stop_recording()
        elif cmd == "o":
            print(get_status())
        elif cmd == "q":
            break
        else:
            print("未知命令")