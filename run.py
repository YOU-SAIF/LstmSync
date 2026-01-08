import argparse
import lstmsync_func
import os

if __name__ == '__main__':
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="LstmSync Inference with Custom Arguments")

    # --- Model & Hardware Settings ---
    parser.add_argument("--human_path", type=str, default="./checkpoints/384.pth", help="Path to model weights (e.g., 384.pth)")
    parser.add_argument("--hubert_path", type=str, default="./checkpoints/chinese-hubert-large", help="Path to Hubert audio encoder")
    parser.add_argument("--key_file", type=str, default="./key.txt", help="Path to validation key file")
    parser.add_argument("--gpu_idx", type=int, default=0, help="GPU Index to use")
    parser.add_argument("--weight_type", type=str, default="fp32", choices=["fp32", "fp16"], help="Precision type")

    # --- Tuning Parameters ---
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--sync_offset", type=int, default=0, help="Audio/Video Sync Offset (frames)")
    parser.add_argument("--scale_h", type=float, default=1.6, help="Mask Height Scale (Increase if cheeks are covered)")
    parser.add_argument("--scale_w", type=float, default=3.6, help="Mask Width Scale (Increase if cheeks are covered)")
    parser.add_argument("--weight_sync", type=float, default=0.5, help="Lip Sync Strength (0.0 - 1.0)")

    # --- Input/Output Paths ---
    parser.add_argument("--video_path", type=str, default="./1.mp4", help="Input video file path")
    parser.add_argument("--audio_path", type=str, default="./1.wav", help="Input audio file path")
    parser.add_argument("--video_out_path", type=str, default="./res.mp4", help="Final output video path")
    
    # --- Temp Files (Optional) ---
    parser.add_argument("--temp_fps25", type=str, default="./fps25_temp.mp4", help="Path for intermediate 25fps video")
    parser.add_argument("--temp_audio", type=str, default="./temp.wav", help="Path for intermediate 16k audio")
    parser.add_argument("--temp_video_prefix", type=str, default="./temp", help="Prefix for temp video generation (no extension)")

    args = parser.parse_args()

    print(f"🚀 Starting LstmSync with: Batch={args.batch_size}, Sync={args.sync_offset}, Mask={args.scale_h}x{args.scale_w}")

    # 2. Initialize Model
    c = lstmsync_func.LstmSync(
        human_path=args.human_path,
        hubert_path=args.hubert_path,
        batch_size=args.batch_size,
        sync_offset=args.sync_offset,
        scale_h=args.scale_h, 
        scale_w=args.scale_w, 
        weight_type=args.weight_type,
        weight_sync=args.weight_sync,
        gpu_idx=args.gpu_idx,
        key_file=args.key_file
    )

    # 3. Run Inference
    out = c.run(
        video_path=args.video_path,
        video_fps25_path=args.temp_fps25,
        video_temp_path=args.temp_video_prefix,
        audio_path=args.audio_path,
        audio_temp_path=args.temp_audio,
        video_out_path=args.video_out_path
    )

    print(out)
