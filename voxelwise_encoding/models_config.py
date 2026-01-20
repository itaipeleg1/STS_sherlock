import numpy as np

models_config_dict = {
    'full': ['amplitude', 'arousal', 'AverME', 'face', 'hue', 'indoor_outdoor', 'mentalization', 'music', 'pitchHz', 'pixel', 'saturation', 'social_nonsocial', 'speaking', 'touch', 'valence', 'written_text'],
    'social': ['social_nonsocial_truncated',"speaking_truncated","valence_truncated","arousal_truncated","mentalization_new"],"social_only":['social_nonsocial_truncated'],'indoor' : ['indoor_outdoor_llava'],'face':['face'],'indoor':['indoor_outdoor'],'landscape':["landscape"],'social_clip': ['social_nonsocial_truncated',"speaking_truncated","valence_truncated","arousal_truncated","mentalization_new",'clip_full'],'speaking_clap': ["speaking","music",'audio_clap'],'social_dino': ['social_nonsocial',"speaking","valence","arousal","mentalization_new",'dino'],'speaking': ["speaking","music"],
    'social_plus_llava': ['arousal', 'mentalization', 'speaking', 'social_nonsocial', 'valence', 'llava_face', 'llava_social', 'llava_touch'],"pc12_clip_clap":['pc12_clip_clap'],"pc12_clip_clap_social":['pc12_clip_clap','social_nonsocial',"speaking","valence","arousal","mentalization_new"],
    'llava_features': ['llava_face', 'llava_social', 'llava_touch'],'500_face':['arousal', 'mentalization', 'speaking', 'social_nonsocial', 'valence'],'500_social':["llava_500days_social_full"],'500_social_speak-gaze':["llava_500days_social_gaze-speak"],
    'C4_social':["llava_social_C41TR"],'C4_social_speak':["llava_social_speak_C41TR"],'C4_social_2TR':["llava_social_C42TR"],'C4_social_speak_2TR':["llava_social_speak_C42TR"],
    'llava_only_social': ['llava_social_speak_full'],'llava_video_3s':["llava_3s_video_results_primitives"],'llava_video_6s':["llava_6s_video_results_primitives"],
    "c4":["annotation_c4"], 'llava_1TR_onlysocial' : ['llava_social'],'llava_2TR_onlysocial' : ['llava_pics_social_non_social(TR2)'],
    'llava_3TR_onlysocial' : ['llava_pics_social_non_social(TR3)'],'llava_4TR_onlysocial' : ['llava_pics_social_non_social(TR4)'],'llava_5TR_onlysocial' : ['llava_pics_social_non_social(TR5)'],
    'llava_6TR_onlysocial' : ['llava_pics_social_non_social(TR6)'],'llava_7TR_onlysocial' : ['llava_pics_social_non_social(TR7)'],
    'llava_8TR_onlysocial' : ['llava_pics_social_non_social(TR8)'],'llava_9TR_onlysocial' : ['llava_pics_social_non_social(TR9)'],
    'llava_10TR_onlysocial' : ['llava_pics_social_non_social(TR10)'],'llava_11TR_onlysocial' : ['llava_pics_social_non_social(TR11)'],
    'llava_12TR_onlysocial' : ['llava_pics_social_non_social(TR12)'],'llava_13TR_onlysocial' : ['llava_pics_social_non_social(TR13)'],'llava_14TR_onlysocial' : ['llava_pics_social_non_social(TR14)'],"llava_15TR_onlysocial" : ['llava_pics_social_non_social(TR15)'],'social_clap': ['social_nonsocial_truncated',"speaking_truncated","valence_truncated","arousal_truncated","mentalization_new",'clap_full'],
    'llava_16TR_onlysocial' : ['llava_pics_social_non_social(TR16)'],'llava_17TR_onlysocial' : ['llava_pics_social_non_social(TR17)'], 'llava_18TR_onlysocial' : ['llava_pics_social_non_social(TR18)'],
    'llava_19TR_onlysocial' : ['llava_pics_social_non_social(TR19)'],'llava_20TR_onlysocial' : ['llava_pics_social_non_social(TR20)'],
    'llava_1TR_video' : ["llava_1.5s_video_new"],'llava_2TR_video' :["llava_3.0s_video"], "llava_3TR_video":["llava_4.5s_video"], "llava_4TR_video":["llava_6.0s_video"],
    'llava_5TR_video' :["llava_7.5s_video"],'llava_6TR_video' :["llava_9.0s_video"],'llava_7TR_video' :["llava_10.5s_video"],
    'llava_8TR_video' :["llava_12.0s_video"],'llava_9TR_video' :["llava_13.5s_video"],
      "llava_music":["social_speak_music"],"cls":["cls_mat_pca"],"vgg":["vgg_mat_pca"],"dino_full":["dino_full"],
     'llava_only_face': ['face_llava_prob(0.4)'], "cls_face":["cls_face_pca"],"cls_social":["CLS_social_pca"],"cls_indoor":["cls_indoor_pca"],"llava_layer16":["llava_social_layer16"],"clap_audio":["audio_clap"],"clip_clap":["audio_clap","clip_pca"],"llava_object_layer16":["llava_object_layer16"],"dino":["dino"],"cls_llava":["cls_pca"],"clip":["clip_pca"],'clip_dino':["clip_pca","dino"],'dino_clap':['dino','audio_clap'],'cls_social_layer25':["cls_social_layer25_version2"],'cls_social_layer25_min':["cls_social_layer25_min"],'unique_variance_social':["cls_social_layer25","clip_pca"],'unique_variance_social2':["cls_social_layer25_5toptokens_40tokens","CLS_social_pca"],'clip_pca1':["clip_pca1"],"cls_pca1":["cls_pca1"],"clip_full" :['clip_full'],"clap_full":['clap_full'],
    'llava_only_face_filled': ['llava_face_filled'],"cls_face_pca1":["cls_face_pca1"],"cls_inside_pca1":["cls_inside_pca1"],"sts_projections_pc1":["STS_projections_pc1"],"sts_projections_pc2":["STS_projections_pc2"],"sts_projections_pc3":["STS_projections_pc3"],"clip_random":["clip_pca_randomized"],"clap_random":["audio_clap_randomized"],
    "leyla_face": ['face'],"llava_logits": ['llava_social_logits'],"clip_llava": ["clip_pca","cls_pca"],"social_llava_llava": ["CLS_social_pca","cls_pca"],"llava_face_llava":["cls_face_pca","cls_pca"],"pca1_clip":["pc1_clip"]
}

