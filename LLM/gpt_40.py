import os
from openai import OpenAI
import base64
import mimetypes
from PIL import Image
import pandas as pd
import argparse

def encode_image_to_base64(path):
    with open(path, "rb") as img_f:
        return base64.b64encode(img_f.read()).decode("utf-8")

def extract_frame_number(filepath):
    try:
        filename = os.path.basename(filepath)
        number_part = filename.split("_")[-1].split(".")[0]
        clean_number = ''.join(c for c in number_part if c.isdigit())
        return int(clean_number)
    except (ValueError, IndexError):
        return -1

def analyze_frames(
    root_dir,
    samples_per_seq,
    seq_prefix,
    seq_range,
    file_extension,
    output_path,
    save_interval
):
    results = []

    # Get TR directories
    seq_dirs = [d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d)) and d.startswith(seq_prefix)]
    seq_dirs.sort(key=lambda x: int(x[len(seq_prefix):]))

    # Apply TR range
    if seq_range:
        start, end = seq_range
        seq_dirs = [d for d in seq_dirs
                    if start <= int(d[len(seq_prefix):]) <= end]

    # Prompt for the model
    prompt_text = (
        "This is a frame from a movie. Describe the interaction between people. Watch for gaze, are the people talking or interacting "
        "in other way and answer. Is there social interaction in this image?"
    )

    for i, seq_dir in enumerate(seq_dirs):
        seq_path = os.path.join(root_dir, seq_dir)
        frame_paths = [os.path.join(seq_path, f) for f in os.listdir(seq_path)
                       if f.endswith(file_extension)]
        frame_paths.sort(key=extract_frame_number)

        if len(frame_paths) < samples_per_seq:
            continue

        indices = [
            j * (len(frame_paths) - 1) // (samples_per_seq - 1)
            for j in range(samples_per_seq)
        ]
        sampled_frames = [frame_paths[j] for j in indices]

        for frame_path in sampled_frames:
            try:
                image_b64 = encode_image_to_base64(frame_path)
                mime_type = mimetypes.guess_type(frame_path)[0] or "image/jpeg"

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}"}
                        }
                    ]
                }]

                client = OpenAI(api_key=args.api_key or os.getenv("OPENAI_API_KEY"))

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=200
                )

                answer = response.choices[0].message.content.strip()
                print(f"[INFO] Processed {frame_path}: {answer}")
            except Exception as e:
                print(f"Error processing {frame_path}: {e}")
                answer = "error"

            results.append({
                "TR": seq_dir,
                "frame": os.path.basename(frame_path),
                "response": answer
            })

        if len(results) % save_interval == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)
            print(f"[INFO] Saved intermediate results to {output_path}")

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved final results to {output_path}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze movie frames with GPT-4o")
    parser.add_argument('--TR_root', type=str, help='Root directory containing TR folders')
    parser.add_argument('--output_path', type=str,  help='Path to save final results CSV')
    parser.add_argument('--start_seq', type=int, default=0, help='Starting TR index (inclusive)')
    parser.add_argument('--end_seq', type=int, help='Ending TR index (inclusive)')
    parser.add_argument('--samples_per_seq', type=int, default=13, help='Number of frames to sample per TR')
    parser.add_argument('--file_extension', type=str, default=".jpg", help='Frame file extension')
    parser.add_argument('--seq_prefix', type=str, default="TR", help='Prefix of TR directories')
    parser.add_argument('--save_interval', type=int, default=50, help='Save results every N frames')
    parser.add_argument('--api_key', type=str, help='OpenAI API key (optional if using env var)')

    args = parser.parse_args()

    # Set API key
    OpenAI.api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not OpenAI.api_key:
        raise ValueError("No API key provided. Use --api_key or set OPENAI_API_KEY in the environment.")

    # Run
    output_path = f"/home/new_storage/sherlock/data/annotations_from_models/chatgpt_4o_responses_social.csv"
    TR_root  = r"/home/new_storage/sherlock/data/frames"
    samples_per_seq = 6
    seq_prefix = "TR"
    seq_range = (0, 250)
    analyze_frames(
        root_dir=TR_root,
        samples_per_seq=samples_per_seq,
        seq_prefix=seq_prefix,
        seq_range=seq_range,
        file_extension='jpg',
        output_path=output_path,
        save_interval=10
    )
