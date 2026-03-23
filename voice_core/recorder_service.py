# voice_core/recorder_service.py
import os
import io
import wave
import threading
import time
from typing import Optional, Callable, Dict, Any
from datetime import datetime
import pyaudio
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class RecorderService:
    """
    录音服务组件 - 支持开始/停止/超时控制，线程安全，可被前端调用
    """

    def __init__(
            self,
            sample_rate: int = 16000,
            channels: int = 1,
            chunk: int = 1024,
            max_duration: int = 60,  # 最大录音时长（秒）
            format: int = pyaudio.paInt16
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.max_duration = max_duration
        self.format = format

        self._audio = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        self._is_recording = False
        self._stop_event = threading.Event()
        self._timeout_thread: Optional[threading.Thread] = None
        self._audio_buffer = io.BytesIO()
        self._wf: Optional[wave.Wave_write] = None

        # 回调函数（可选）
        self.on_start: Optional[Callable] = None
        self.on_stop: Optional[Callable[[bytes], None]] = None
        self.on_timeout: Optional[Callable] = None
        self.on_error: Optional[Callable[[Exception], None]] = None

    def start(self, callback: Optional[Callable[[bytes], None]] = None) -> Dict[str, Any]:
        """
        开始录音
        :param callback: 录音结束后的回调函数，接收音频二进制数据
        :return: 状态信息
        """
        if self._is_recording:
            return {"success": False, "message": "Already recording"}

        try:
            # 重置状态
            self._stop_event.clear()
            self._audio_buffer = io.BytesIO()
            self._wf = wave.open(self._audio_buffer, 'wb')
            self._wf.setnchannels(self.channels)
            self._wf.setsampwidth(self._audio.get_sample_size(self.format))
            self._wf.setframerate(self.sample_rate)

            # 打开音频流
            self._stream = self._audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk
            )

            self._is_recording = True
            self._callback = callback

            # 启动录音线程
            threading.Thread(target=self._record_loop, daemon=True).start()

            # 启动超时监控
            self._timeout_thread = threading.Thread(
                target=self._timeout_monitor,
                daemon=True
            )
            self._timeout_thread.start()

            if self.on_start:
                self.on_start()

            return {
                "success": True,
                "message": "Recording started",
                "start_time": datetime.now().isoformat(),
                "max_duration": self.max_duration
            }

        except Exception as e:
            self._cleanup()
            if self.on_error:
                self.on_error(e)
            return {"success": False, "message": f"Start failed: {str(e)}"}

    def stop(self) -> Dict[str, Any]:
        """手动停止录音"""
        if not self._is_recording:
            return {"success": False, "message": "Not recording"}

        self._stop_event.set()
        return {"success": True, "message": "Stopping..."}

    def is_recording(self) -> bool:
        """查询录音状态"""
        return self._is_recording

    def _record_loop(self):
        """录音主循环（运行在独立线程）"""
        try:
            while not self._stop_event.is_set() and self._stream.is_active():
                data = self._stream.read(self.chunk, exception_on_overflow=False)
                if self._wf and self._is_recording:
                    self._wf.writeframes(data)
            self._finish_recording()
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            self._cleanup()

    def _timeout_monitor(self):
        """超时监控线程"""
        start_time = time.time()
        while self._is_recording and not self._stop_event.is_set():
            if time.time() - start_time >= self.max_duration:
                print(f"⏰ 录音已达 {self.max_duration} 秒，自动停止")
                self._stop_event.set()
                if self.on_timeout:
                    self.on_timeout()
                break
            time.sleep(0.5)

    def _finish_recording(self):
        """录音结束处理"""
        self._is_recording = False
        self._cleanup()

        # 获取音频数据
        audio_data = self._audio_buffer.getvalue()

        # 调用回调（如：上传 ASR）
        if self._callback:
            try:
                self._callback(audio_data)
            except Exception as e:
                print(f"❌ Callback error: {e}")
                if self.on_error:
                    self.on_error(e)

        if self.on_stop:
            self.on_stop(audio_data)

    def _cleanup(self):
        """资源清理"""
        if self._wf:
            self._wf.close()
            self._wf = None
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        # 注意：不要 close self._audio，避免重复初始化开销

    def __del__(self):
        """析构时确保资源释放"""
        if self._is_recording:
            self.stop()
            time.sleep(0.1)  # 等待线程退出
        if self._audio:
            self._audio.terminate()