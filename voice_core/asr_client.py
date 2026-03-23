# voice_core/asr_client.py
import os
import io
import wave
import dashscope
from dashscope import MultiModalConversation
from dotenv import load_dotenv

load_dotenv()


class ASRClient:
    """DashScope ASR 客户端封装"""

    def __init__(
            self,
            api_key: str = None,
            model: str = "qwen3-asr-flash",
            base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip('/')

        if not self.api_key:
            raise ValueError("API Key is required")

        dashscope.api_key = self.api_key
        dashscope.base_http_api_url = self.base_url

    def transcribe_bytes(self, audio_bytes: bytes, language: str = "zh") -> dict:
        """
        转录音频字节流
        :param audio_bytes: WAV 格式音频二进制数据
        :param language: 语种（zh/en 等）
        :return: ASR 响应结果
        """
        # 注意：DashScope 多模态接口目前更支持 URL 或本地文件路径
        # 方案：将 bytes 写入临时文件
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            messages = [
                {"role": "user", "content": [{"audio": tmp_path}]}
            ]

            response = MultiModalConversation.call(
                model=self.model,
                messages=messages,
                result_format="message",
                asr_options={
                    "language": language,
                    "enable_itn": False
                }
            )
            return response
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def transcribe_file(self, file_path: str, language: str = "zh") -> dict:
        """直接转录本地音频文件"""
        messages = [
            {"role": "user", "content": [{"audio": file_path}]}
        ]

        return MultiModalConversation.call(
            model=self.model,
            messages=messages,
            result_format="message",
            asr_options={
                "language": language,
                "enable_itn": False
            }
        )