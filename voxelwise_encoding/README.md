## Voxelwise Encoding Analysis

### Overview
 Ridge Regression for voxelwise encoding analysis of fMRI data, correlating brain activity with various features extracted from video content.


### Input Requirements

#### fMRI Data
- **Path Structure**: `fmri_data_path/sub{n}/derivatives/sherlock_movie_s{subj}.nii`
- **File Format**: 4D Nifti file
- **Origin**: Sherlock's Fmri scannings were downloaded from here: https://dataspace.princeton.edu/handle/88435/dsp01nz8062179

#### Feature Annotations
- **Path**: `annotations_path/`
- **Format**: Individual .npy files for each feature
- **Requirements**: Array length must match fMRI data TRs, If not (experiment for example) Cut FMRi length from args
- **Example Files**:
  - faces.npy
  - objects.npy
  - llava_social_speak.npy

#### Mask (Optional)

- **Format**: 3D Nifti file
- **Note**: Without mask, whole brain analysis will be performed (longer processing time)

#### Trials (optional)

- **Note**: This option was added recently to investigate the influence of moving average on annotation and it's relation to the Linear model
