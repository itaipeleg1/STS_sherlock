import numpy as np

models_config_dict = {
    'full': ['amplitude', 'arousal', 'AverME', 'face', 'hue', 'indoor_outdoor', 'mentalization', 'music', 'pitchHz', 'pixel', 'saturation', 'social_nonsocial', 'speaking', 'touch', 'valence', 'written_text'],
    'social': ['social_nonsocial'],
    'social_plus_llava': ['arousal', 'mentalization', 'speaking', 'social_nonsocial', 'valence', 'llava_face', 'llava_social', 'llava_touch'],
    'llava_features': ['llava_face', 'llava_social', 'llava_touch'],
    ## This is the correct annotation for the llava social features
    'llava_only_social': ['llava_social_speak_full'],
    
      "llava_music":["social_speak_music"],
     'llava_only_face': ['llava_face'],
    'llava_only_face_filled': ['llava_face_filled'],
    "leyla_face": ['face'],
}

