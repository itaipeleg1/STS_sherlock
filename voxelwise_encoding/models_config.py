import numpy as np

models_config_dict = {
    'full': ['amplitude', 'arousal', 'AverME', 'face', 'hue', 'indoor_outdoor', 'mentalization', 'music', 'pitchHz', 'pixel', 'saturation', 'social_nonsocial', 'speaking', 'touch', 'valence', 'written_text'],
    'social': ['social_nonsocial'],
    'social_plus_llava': ['arousal', 'mentalization', 'speaking', 'social_nonsocial', 'valence', 'llava_face', 'llava_social', 'llava_touch'],
    'llava_features': ['llava_face', 'llava_social', 'llava_touch'],
    ## This is the correct annotation for the llava social features
    'llava_only_social': ['llava_social_speak_full'],'llava_video_3s':["llava_3s_video_results_primitives"],'llava_video_6s':["llava_6s_video_results_primitives"],
    "c4":["annotation_c4"], 'llava_1TR_onlysocial' : ['llava_pics_social_non_social(TR1)'],'llava_2TR_onlysocial' : ['llava_pics_social_non_social(TR2)'],
    'llava_3TR_onlysocial' : ['llava_pics_social_non_social(TR3)'],'llava_4TR_onlysocial' : ['llava_pics_social_non_social(TR4)'],'llava_5TR_onlysocial' : ['llava_pics_social_non_social(TR5)'],
    'llava_6TR_onlysocial' : ['llava_pics_social_non_social(TR6)'],'llava_7TR_onlysocial' : ['llava_pics_social_non_social(TR7)'],
    'llava_8TR_onlysocial' : ['llava_pics_social_non_social(TR8)'],'llava_9TR_onlysocial' : ['llava_pics_social_non_social(TR9)'],
    'llava_10TR_onlysocial' : ['llava_pics_social_non_social(TR10)'],
      "llava_music":["social_speak_music"],
     'llava_only_face': ['llava_face'],"cls":["cls"],
    'llava_only_face_filled': ['llava_face_filled'],
    "leyla_face": ['face'],
}

