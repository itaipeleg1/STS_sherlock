# STS_sherlock
# Automated Social Interaction Detection in Sherlock Using LLAVA

## Project Overview
This project aims to replicate and extend the findings from Masson & Isik (2021), who demonstrated functional selectivity for social interaction perception in the human superior temporal sulcus (STS) during natural viewing. While the original study relied on manual labeling of social interactions in the Sherlock TV series, this project leverages LLAVA (Large Language and Vision Assistant) to automate the detection process.

## Original Research
The original study by Masson & Isik (2021), published in NeuroImage, analyzed how the brain processes social interactions during natural viewing conditions. They specifically:
- Manually labeled social interactions in episodes of Sherlock
- Identified correlations between these interactions and brain activation in the STS region using fMRI
- [Original paper link](https://www.sciencedirect.com/science/article/pii/S1053811921010132)

## Project Goals
This project seeks to:
1. Automate the detection of social interactions using LLAVA
2. Compare automated detection results with the manual annotations from the original study
3. Evaluate the feasibility of using AI for large-scale social interaction labeling in video content

## Methodology
- Using LLAVA for automated detection of social interactions
- Processing Sherlock TV show episodes
- Comparing AI-generated labels with original manual annotations

## Repository Structure
```voxelwise_encoding/
├── voxelwise_encoding_ridge.
├── utils
├── model_config
└── ..

### This project is based on Shiri almog's work : https://github.com/Shirialmog/STS