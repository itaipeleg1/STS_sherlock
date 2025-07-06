import cv2
import os
import argparse 

def extract_frames(video_path, output_dir, TR=1):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video FPS: {fps}")
    print(f"Total frames in video: {total_frames}")
    
    
    frame_count = 0
    current_tr = 0
    frames_in_current_tr = 0
    
    frames_per_tr = int(round(fps * TR))  # e.g., 25 frames for TR=1 if fps=25

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create TR-specific subfolder
        tr_dir = os.path.join(output_dir, f"TR{current_tr:04d}")
        os.makedirs(tr_dir, exist_ok=True)

        # Save frame in TR subfolder
        frame_path = os.path.join(tr_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_count += 1
        frames_in_current_tr += 1

        if frames_in_current_tr >= frames_per_tr:
            current_tr += 1
            frames_in_current_tr = 0
            print(f"TR transition at frame {frame_count}")


        if frame_count % 1000 == 0:
            print(f"Processed {frame_count} frames")
    cap.release()
    print(f"Finished! Extracted {frame_count} frames")
    print(f"Total TRs processed: {current_tr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cut the movie into frames')

    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to input movie file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output frames')
   

    args = parser.parse_args()

    extract_frames(args.video_path, args.output_dir)