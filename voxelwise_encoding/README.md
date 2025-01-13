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

#### ISC Mask (Optional)
Currently not updated in this repo
- **Path**: `isc_mask_path/isc_mask.nii.gz`
- **Format**: 3D Nifti file
- **Note**: Without mask, whole brain analysis will be performed (longer processing time)
