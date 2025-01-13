import cv2
import argparse
import scipy.io
import os
import numpy as np

def get_annotations(ann_dir, type):
    annotation_file = f"{ann_dir}\\{type}.mat"
    mat = scipy.io.loadmat(annotation_file)
    key_name = annotation_file.split("\\")[-1].split(".")[0]
    try:
        data = mat[key_name]
        s1 = data[27:946]
        s2 = data[946+28:]
        data = np.concatenate([s1, s2])
    except:
        print ('Error: missing annotation file for ', annotation_file)
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
    target_fps = 24.0
    
    print(f"Original FPS: {original_fps}, Target FPS: {target_fps}")
    
    # Create output video directly at 24 fps
    out_cap = cv2.VideoWriter(new_path, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (width, height))
    
    tr = 1.5
    frames_per_tr = int(target_fps * tr) 
    frame_count = 0
    tr_delay = 3
    frame_delay = int(tr * 3 * target_fps)
    len_annotations = all_annotations["face"].shape[0]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        current_tr = frame_count // frames_per_tr
            
        if frame_count > frame_delay:
            text = add_relevant_annotation(all_annotations, current_tr-tr_delay)
        else:
            text = "4.5 sec delay due to hemodynamic lag"
            
        timing_text = f"TR: {current_tr}/{len_annotations}, Frame: {frame_count}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, timing_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('frame', frame)
        out_cap.write(frame)
        
        if cv2.waitKey(int(1000/24)) & 0xFF == ord('q'):
            break
            
    cap.release()
    out_cap.release()
    cv2.destroyAllWindows()
    # Add this part:
    import subprocess
    print("Fixing video format...")
    fixed_path = 'fixed_' + os.path.basename(new_path)
    subprocess.run(['ffmpeg', '-i', new_path, '-c', 'copy', fixed_path])
    print(f"Fixed video saved as: {fixed_path}")


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
    written_text_annotations = get_annotations(annotation_dir, "written_text")
    socia_noosocial_annotations = get_annotations(annotation_dir, "social_nonsocial")
    speaking_annotations = get_annotations(annotation_dir, "speaking")
    indoor_outdoor_annotations = get_annotations(annotation_dir, "indoor_outdoor")

    all_annotations = {"face": faces_annotations,
                       "written_text": written_text_annotations,
                       "social_nonsocial": socia_noosocial_annotations,
                       "speaking": speaking_annotations,
                       "indoor_outdoor": indoor_outdoor_annotations}

    add_text_to_movie(movie_path, new_path, all_annotations)
