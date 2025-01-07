import numpy as np

models_config_dict = {
    'full': ['amplitude', 'arousal', 'AverME', 'face', 'hue', 'indoor_outdoor', 'mentalization', 'music', 'pitchHz', 'pixel', 'saturation', 'social_nonsocial', 'speaking', 'touch', 'valence', 'written_text'],
    'social': ['social_nonsocial2'],
    'social_plus_llava': ['arousal', 'mentalization', 'speaking', 'social_nonsocial', 'valence', 'llava_face', 'llava_social', 'llava_touch'],
    'llava_features': ['llava_face', 'llava_social', 'llava_touch'],
    'llava_only_social': ['social_speak2']
    , 'llava_only_face': ['llava_face'],
    'llava_only_face_filled': ['llava_face_filled'],
    "leyla_face": ['face'],
}

