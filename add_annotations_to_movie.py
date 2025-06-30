import cv2
import argparse
import os
import numpy as np

def get_annotations(ann_dir, type):
    # Use np.load for .npy files and os.path.join for paths
    annotation_file = os.path.join(ann_dir, f"{type}.npy")
    try:
        data = np.load(annotation_file)
        # If you need to slice or process, do it here (otherwise just return data)
       # data = data[25:]
        s1 = data[25:946]
        s2 = data[946+25:]
        data = np.concatenate([s1, s2])
    except Exception as e:
        print('Error: missing annotation file for', annotation_file, e)
        data = None
    return data


def add_relevant_annotation(all_annotations, count_annotation_frames):
    text = ''
    for key in all_annotations:
        if all_annotations[key][count_annotation_frames] == 1:
            text += f'{key}, '
        else:
            text += f'no {key}, '
    return text


def add_text_to_movie(movie_path, new_path, all_annotations):
    cap = cv2.VideoCapture(movie_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 24

    print(f"Original FPS: {original_fps}, Target FPS: {target_fps}")

    out_cap = cv2.VideoWriter(new_path, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (width, height))

    tr = 1.5
    frames_per_tr = int(target_fps * tr)
    frame_count = 0
    output_frame_count = 0  # Track output frames for TR calculation
    len_annotations = all_annotations["face"].shape[0]

    # Get total frame count for progress reporting
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_tr = output_frame_count // frames_per_tr
        if current_tr >= len_annotations:
            current_tr = len_annotations - 1

        text = add_relevant_annotation(all_annotations, current_tr)
        timing_text = f"TR: {current_tr+1}/{len_annotations}, Frame: {frame_count}, Output: {output_frame_count}"

        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, timing_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out_cap.write(frame)

        # Progress print only (no release/reopen)
        if frame_count % 250 == 0 and frame_count > 0:
            print(f"Progress: {frame_count}/{total_frames} frames processed and saved.")

        frame_count += 1
        output_frame_count += 1

    cap.release()
    out_cap.release()
    cv2.destroyAllWindows()
    # Add this part:
    import subprocess
    print("Fixing video format...")
    fixed_path = os.path.join(os.path.dirname(new_path), 'fixed_' + os.path.basename(new_path) + '.mp4')
    result = subprocess.run(['ffmpeg', '-y', '-i', new_path, '-c', 'copy', fixed_path], capture_output=True)
    if result.returncode == 0:
        print(f"Fixed video saved as: {fixed_path}")
    else:
        print("ffmpeg error:", result.stderr.decode())


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--annotation_dir", type=str, required=True, help='The directory for the annotations')
    args.add_argument("--movie_path", type=str, required=True,help="path for cutted movie" )
    args.add_argument("--new_path", type=str, required=True,help="where to save the movie with annotations")
    args = args.parse_args()

    annotation_dir = args.annotation_dir
    movie_path = args.movie_path
    new_path = args.new_path

    faces_annotations = get_annotations(annotation_dir, "face")
    socia_noosocial_annotations = get_annotations(annotation_dir, "social_nonsocial")
    indoor_outdoor_annotations = get_annotations(annotation_dir, "indoor_outdoor")

    all_annotations = {"face": faces_annotations,
                       "social_nonsocial": socia_noosocial_annotations,
                       "indoor_outdoor": indoor_outdoor_annotations}

    add_text_to_movie(movie_path, new_path, all_annotations)
