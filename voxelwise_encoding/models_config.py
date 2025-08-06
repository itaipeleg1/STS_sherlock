import numpy as np

models_config_dict = {
    'full': ['amplitude', 'arousal', 'AverME', 'face', 'hue', 'indoor_outdoor', 'mentalization', 'music', 'pitchHz', 'pixel', 'saturation', 'social_nonsocial', 'speaking', 'touch', 'valence', 'written_text'],
    'social': ['social_nonsocial'],'indoor' : ['indoor_outdoor_llava'],
    'social_plus_llava': ['arousal', 'mentalization', 'speaking', 'social_nonsocial', 'valence', 'llava_face', 'llava_social', 'llava_touch'],
    'llava_features': ['llava_face', 'llava_social', 'llava_touch'],'500_face':['arousal', 'mentalization', 'speaking', 'social_nonsocial', 'valence'],'500_social':["llava_500days_social_full"],'500_social_speak-gaze':["llava_500days_social_gaze-speak"],
    'C4_social':["llava_social_C41TR"],'C4_social_speak':["llava_social_speak_C41TR"],'C4_social_2TR':["llava_social_C42TR"],'C4_social_speak_2TR':["llava_social_speak_C42TR"],
    'llava_only_social': ['llava_social_speak_full'],'llava_video_3s':["llava_3s_video_results_primitives"],'llava_video_6s':["llava_6s_video_results_primitives"],
    "c4":["annotation_c4"], 'llava_1TR_onlysocial' : ['llava_social'],'llava_2TR_onlysocial' : ['llava_pics_social_non_social(TR2)'],
    'llava_3TR_onlysocial' : ['llava_pics_social_non_social(TR3)'],'llava_4TR_onlysocial' : ['llava_pics_social_non_social(TR4)'],'llava_5TR_onlysocial' : ['llava_pics_social_non_social(TR5)'],
    'llava_6TR_onlysocial' : ['llava_pics_social_non_social(TR6)'],'llava_7TR_onlysocial' : ['llava_pics_social_non_social(TR7)'],
    'llava_8TR_onlysocial' : ['llava_pics_social_non_social(TR8)'],'llava_9TR_onlysocial' : ['llava_pics_social_non_social(TR9)'],
    'llava_10TR_onlysocial' : ['llava_pics_social_non_social(TR10)'],'llava_11TR_onlysocial' : ['llava_pics_social_non_social(TR11)'],
    'llava_12TR_onlysocial' : ['llava_pics_social_non_social(TR12)'],'llava_13TR_onlysocial' : ['llava_pics_social_non_social(TR13)'],'llava_14TR_onlysocial' : ['llava_pics_social_non_social(TR14)'],"llava_15TR_onlysocial" : ['llava_pics_social_non_social(TR15)'],
    'llava_16TR_onlysocial' : ['llava_pics_social_non_social(TR16)'],'llava_17TR_onlysocial' : ['llava_pics_social_non_social(TR17)'], 'llava_18TR_onlysocial' : ['llava_pics_social_non_social(TR18)'],
    'llava_19TR_onlysocial' : ['llava_pics_social_non_social(TR19)'],'llava_20TR_onlysocial' : ['llava_pics_social_non_social(TR20)'],
    'llava_1TR_video' : ["llava_1.5s_video_new"],'llava_2TR_video' :["llava_3.0s_video"], "llava_3TR_video":["llava_4.5s_video"], "llava_4TR_video":["llava_6.0s_video"],
    'llava_5TR_video' :["llava_7.5s_video"],'llava_6TR_video' :["llava_9.0s_video"],'llava_7TR_video' :["llava_10.5s_video"],
    'llava_8TR_video' :["llava_12.0s_video"],'llava_9TR_video' :["llava_13.5s_video"],
      "llava_music":["social_speak_music"],"cls":["cls_mat_pca"],"vgg":["vgg_mat_pca"],
     'llava_only_face': ['face_llava_prob(0.4)'], "cls":["cls_mat_pca"],
    'llava_only_face_filled': ['llava_face_filled'],
    "leyla_face": ['face'],"llava_logits": ['llava_social_logits']
}

