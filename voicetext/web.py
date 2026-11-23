"""Веб-интерфейс (Gradio): загрузите видео — получите текст, SRT, VTT и JSON."""

from __future__ import annotations

import tempfile


def build_app(checkpoint: str, device: str = "auto"):
    try:
        import gradio as gr
    except ImportError as e:
        raise SystemExit("Для веб-интерфейса установите gradio: pip install -r requirements-web.txt") from e

    from voicetext.transcribe import Transcriber, save_result

    transcriber = Transcriber(checkpoint, device)
    out_root = tempfile.mkdtemp(prefix="voicetext_")

    def run(video_path, formats, audio_track):
        if not video_path:
            raise gr.Error("Загрузите видео или аудио файл")
        try:
            result = transcriber.transcribe_file(video_path, audio_track=(audio_track or "").strip() or None)
        except RuntimeError as e:
            raise gr.Error(str(e))
        files = save_result(result, video_path, out_root, formats or ["json", "srt", "vtt"])
        lang = result["language"] or "не определён"
        info = f"Язык: {lang} · длительность: {result['duration']:.1f} с · фраз: {len(result['segments'])}"
        return result["text"], info, files

    with gr.Blocks(title="VoiceText — текст из видео") as app:
        gr.Markdown("## VoiceText — извлечение текста из видео по озвучке\n"
                    f"Модель: `{transcriber.checkpoint}` · устройство: `{transcriber.device}`")
        with gr.Row():
            with gr.Column():
                video = gr.File(label="Видео или аудио", type="filepath")
                formats = gr.CheckboxGroup(["json", "srt", "vtt", "txt"], value=["json", "srt", "vtt"],
                                           label="Форматы результата")
                audio_track = gr.Textbox(label="Звуковая дорожка (необязательно)",
                                         placeholder="номер (0, 1…) или язык: ru, en")
                btn = gr.Button("Распознать", variant="primary")
            with gr.Column():
                text = gr.Textbox(label="Текст", lines=12)
                info = gr.Markdown()
                files = gr.File(label="Файлы", file_count="multiple")
        btn.click(run, [video, formats, audio_track], [text, info, files])
    return app


def serve(checkpoint: str, host: str = "127.0.0.1", port: int = 7860, device: str = "auto", share: bool = False):
    build_app(checkpoint, device).launch(server_name=host, server_port=port, share=share)
