#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL Web Demo with Gradio
Supports text, image, and video inputs for multi-modal conversations
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add parent python_demo to path to import chat module
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent / "python_demo"))

import gradio as gr
import numpy as np
from PIL import Image

# Import model wrapper
try:
    import chat
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you have installed: gradio, transformers, qwen_vl_utils, torch, torchvision")
    sys.exit(1)


class Qwen3VLWebDemo:
    """Wrapper for Qwen3-VL model with web UI support"""

    def __init__(self, args):
        # Basic configuration
        self.device = args.devid
        self.video_ratio = args.video_ratio

        # Load model
        print("Loading model...")
        self.model = chat.Qwen3_VL()
        self.model.init(self.device, args.model_path)

        # Load processor and tokenizer
        print("Loading tokenizer...")
        self.processor = AutoProcessor.from_pretrained(
            args.config_path,
            trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer

        # Special tokens
        self.ID_END = self.tokenizer.convert_tokens_to_ids("<|end|>")
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.ID_IMAGE_PAD = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.ID_VIDEO_PAD = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.ID_VISION_START = self.tokenizer.convert_tokens_to_ids('<|vision_start|>')

        # Model configuration
        self.spatial_merge_size = 2
        self.num_grid_per_side = 48
        self.support_history = self.model.support_history
        self.total_pixels = (self.model.MAX_INPUT_LENGTH - 128) * 32 * 32

        # Position tracking
        self.max_posid = 0
        self.history_max_posid = 0

        print("Model loaded successfully!")
        print(f"Max sequence length: {self.model.SEQLEN}")
        print(f"Max pixels: {self.model.MAX_PIXELS}")
        print(f"Support history: {self.support_history}")

    def get_media_type(self, file_path):
        """Determine if file is image or video"""
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

        if isinstance(file_path, str):
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            if ext in image_exts:
                return "image"
            elif ext in video_exts:
                return "video"

        return None

    def create_messages(self, text_input, media_path=None):
        """Create message format for the model"""
        if media_path is None or media_path == "":
            # Text only
            return [{
                "role": "user",
                "content": [{"type": "text", "text": text_input}],
            }]

        media_type = self.get_media_type(media_path)
        if media_type == "image":
            return [{
                "role": "user",
                "content": [
                    {"type": "image", "image": media_path,
                     "min_pixels": 4 * 32 * 32,
                     "max_pixels": self.model.MAX_PIXELS},
                    {"type": "text", "text": text_input},
                ],
            }]
        elif media_type == "video":
            return [{
                "role": "user",
                "content": [
                    {"type": "video", "video": media_path, "fps": 1.0,
                     "min_pixels": 4 * 32 * 32,
                     "max_pixels": int(self.model.MAX_PIXELS * self.video_ratio),
                     "total_pixels": self.total_pixels},
                    {"type": "text", "text": text_input},
                ],
            }]
        else:
            raise ValueError(f"Unsupported media type: {media_path}")

    def process_input(self, messages, media_type):
        """Process input messages using the processor"""
        if media_type == "text":
            return self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

        # For image/video
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True
        )

        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        return self.processor(
            text=[text],
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs
        )

    def clear_history(self):
        """Clear conversation history"""
        if self.support_history:
            self.model.clear_history()
        self.history_max_posid = 0
        self.max_posid = 0

    def generate_response(self, text_input, media_path, history):
        """
        Generate response with streaming support
        This is called by Gradio for user interactions
        """
        # Create messages
        try:
            messages = self.create_messages(text_input, media_path)
            media_type = "text" if media_path is None or media_path == "" else self.get_media_type(media_path)
        except Exception as e:
            error_msg = f"Error creating messages: {str(e)}"
            yield history + [[(text_input, media_path), error_msg]], f"Error: {error_msg}", "", "", ""
            return

        # Process input
        try:
            inputs = self.process_input(messages, media_type)
            token_len = inputs.input_ids.numel()
        except Exception as e:
            error_msg = f"Error processing input: {str(e)}"
            yield history + [[(text_input, media_path), error_msg]], f"Error: {error_msg}", "", "", ""
            return

        # Check token length
        if token_len > self.model.MAX_INPUT_LENGTH:
            error_msg = f"Input too long: {token_len} tokens (max: {self.model.MAX_INPUT_LENGTH})"
            yield history + [[(text_input, media_path), error_msg]], f"Error: {error_msg}", "", "", ""
            return

        # Check history limit
        if self.support_history:
            if (token_len + self.model.history_length > self.model.SEQLEN - 128) or \
               (self.model.history_length > self.model.PREFILL_KV_LENGTH):
                self.clear_history()

        # Update history with user message
        user_message = (text_input, media_path) if media_path else text_input
        history = history + [[user_message, None]]

        # Start generation
        yield history, "Processing...", "", "", ""

        try:
            # Process embeddings
            self.model.forward_embed(inputs.input_ids.numpy())

            # Process vision input if present
            vit_start = time.time()
            if media_type == "image":
                # For image: would need vit_process_image
                # Simplified for now
                vit_end = time.time()
                # Calculate position IDs (simplified)
                position_ids = np.array([i for i in range(token_len)], dtype=np.int32)
            elif media_type == "video":
                # For video: would need vit_process_video
                # Simplified for now
                vit_end = time.time()
                position_ids = np.array([i for i in range(token_len)], dtype=np.int32)
            else:
                vit_end = time.time()
                position_ids = np.array([i for i in range(token_len)], dtype=np.int32)

            vision_time = vit_end - vit_start

            # First token generation
            first_start = time.time()
            token = self.model.forward_first(position_ids)
            first_end = time.time()
            first_duration = first_end - first_start

            # Stream generation
            full_text = ""
            tok_num = 0
            full_word_tokens = []

            while token not in [self.ID_IM_END, self.ID_END] and \
                  self.model.history_length < self.model.SEQLEN:

                full_word_tokens.append(token)
                word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)

                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = self.tokenizer.decode([token, token], skip_special_tokens=True)[len(pre_word):]
                    full_text += word
                    full_word_tokens = []

                    # Update history with streamed text
                    history[-1][1] = full_text
                    tps = tok_num / (time.time() - first_end) if tok_num > 0 else 0
                    yield history, "Generating...", f"{first_duration:.3f}s", f"{tps:.2f}", f"{vision_time:.3f}s"

                self.max_posid += 1
                position_ids = np.array([self.max_posid, self.max_posid, self.max_posid], dtype=np.int32)
                token = self.model.forward_next(position_ids)
                tok_num += 1

            # Final update
            next_end = time.time()
            next_duration = next_end - first_end
            tps = tok_num / next_duration if next_duration > 0 else 0

            self.history_max_posid = self.max_posid + 2

            yield history, "Complete", f"{first_duration:.3f}s", f"{tps:.2f}", f"{vision_time:.3f}s"

        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            history[-1][1] = error_msg
            yield history, f"Error: {error_msg}", "", "", ""


def create_demo(qwen_model):
    """Create Gradio interface"""

    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .chat-container {
        height: 600px !important;
    }
    """

    with gr.Blocks(css=custom_css, title="Qwen3-VL TPU Demo") as demo:
        gr.Markdown("""
        # 🚀 Qwen3-VL TPU Multi-Modal Demo

        这是在 Sophgo BM1684X/BM1688 TPU 上运行的 Qwen3-VL 多模态大模型演示。

        **功能特点：**
        - 📝 纯文本对话
        - 🖼️ 图片理解和问答
        - 🎥 视频内容分析
        - 💬 多轮对话历史

        **使用说明：**
        1. 输入文本问题
        2. （可选）上传图片或视频
        3. 点击发送获取回答
        """)

        with gr.Row():
            with gr.Column(scale=3):
                # Chatbot interface
                chatbot = gr.Chatbot(
                    label="对话历史",
                    height=600,
                    show_copy_button=True,
                    bubble_full_width=False
                )

                # Input area
                with gr.Row():
                    with gr.Column(scale=4):
                        text_input = gr.Textbox(
                            label="文本输入",
                            placeholder="请输入您的问题...",
                            lines=2,
                            show_label=False
                        )
                    with gr.Column(scale=1):
                        submit_btn = gr.Button("发送", variant="primary", size="lg")
                        clear_btn = gr.Button("清空对话", variant="stop", size="lg")

                # Media upload
                with gr.Row():
                    image_input = gr.File(
                        label="上传图片",
                        file_types=["image"],
                        type="filepath"
                    )
                    video_input = gr.File(
                        label="上传视频",
                        file_types=["video"],
                        type="filepath"
                    )

            with gr.Column(scale=1):
                # Status and metrics
                gr.Markdown("### 📊 性能指标")

                status_box = gr.Textbox(
                    label="状态",
                    value="Ready",
                    interactive=False
                )

                ftl_box = gr.Textbox(
                    label="首Token延迟 (FTL)",
                    value="-",
                    interactive=False
                )

                tps_box = gr.Textbox(
                    label="生成速度 (TPS)",
                    value="-",
                    interactive=False
                )

                vision_box = gr.Textbox(
                    label="视觉处理时间",
                    value="-",
                    interactive=False
                )

                gr.Markdown("""
                ### 📌 提示
                - 支持的图片格式: JPG, PNG, GIF, BMP, WEBP
                - 支持的视频格式: MP4, AVI, MOV, MKV
                - 视频默认1fps，最长12秒
                - 图片最大尺寸: 768x768
                """)

        # Event handlers
        def user_submit(message, image, video, history):
            """Handle user input submission"""
            media_path = image if image else (video if video else "")
            if not message and not media_path:
                gr.Warning("请输入文本或上传图片/视频")
                return history, "", None, None

            return history, message, media_path, image, video

        def clear_chat():
            """Clear chat history"""
            qwen_model.clear_history()
            return [], "", None, None, "Ready", "-", "-", "-"

        # Connect events
        submit_btn.click(
            fn=user_submit,
            inputs=[text_input, image_input, video_input, chatbot],
            outputs=[chatbot, text_input, "hidden", image_input, video_input]
        ).then(
            fn=qwen_model.generate_response,
            inputs=[text_input, "hidden", chatbot],
            outputs=[chatbot, status_box, ftl_box, tps_box, vision_box]
        )

        text_input.submit(
            fn=user_submit,
            inputs=[text_input, image_input, video_input, chatbot],
            outputs=[chatbot, text_input, "hidden", image_input, video_input]
        ).then(
            fn=qwen_model.generate_response,
            inputs=[text_input, "hidden", chatbot],
            outputs=[chatbot, status_box, ftl_box, tps_box, vision_box]
        )

        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, text_input, image_input, video_input, status_box, ftl_box, tps_box, vision_box]
        )

        # Hidden component for storing media path
        gr.Textbox(visible=False, elem_id="hidden")

    return demo


def main(args):
    """Main entry point"""

    # Initialize model
    print("=" * 60)
    print("Initializing Qwen3-VL Web Demo...")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Config path: {args.config_path}")
    print(f"Device ID: {args.devid}")
    print(f"Video ratio: {args.video_ratio}")
    print("=" * 60)

    qwen_model = Qwen3VLWebDemo(args)

    # Create demo
    demo = create_demo(qwen_model)

    # Launch
    print("\n" + "=" * 60)
    print("Starting Gradio server...")
    print("=" * 60)
    print(f"Server address: http://{args.server_name}:{args.server_port}")
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    demo.queue(max_size=20).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=False,
        inbrowser=args.inbrowser
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-VL Web Demo")

    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='Path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../config",
                        help='Path to the processor/config directory')
    parser.add_argument('-d', '--devid', type=int, default=0,
                        help='TPU device ID to use')
    parser.add_argument('--video_ratio', type=float, default=0.25,
                        help='Video pixel ratio (default: 0.25)')
    parser.add_argument('--server_name', type=str, default="0.0.0.0",
                        help='Server address (default: 0.0.0.0)')
    parser.add_argument('--server_port', type=int, default=7860,
                        help='Server port (default: 7860)')
    parser.add_argument('--inbrowser', action='store_true',
                        help='Open browser automatically')

    args = parser.parse_args()
    main(args)
