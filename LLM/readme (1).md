# Frame Sequence Analysis with LLaVA

A Python tool for analyzing sequences of image frames using the LLaVA (Large Language and Vision Assistant) model. This tool process sequences of frames to detect features

## Features

- Process sequences of image frames using LLaVA model
- Customizable frame sampling per sequence
- Support for different sequence ranges
- Automatic intermediate saving of results
- Configurable prompts for different types of analysis
- Command-line interface for easy use

## Prerequisites

```bash
pip install -q -U transformers==4.47.0
pip install -q bitsandbytes==0.41.3 accelerate==0.26.0
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--TR_root` | str | Required | Root directory containing TR sequences |
| `--output_path` | str | Required | Path to save results CSV |
| `--start_seq` | int | 0 | Starting sequence number |
| `--end_seq` | int | 1000 | Ending sequence number |
| `--samples_per_seq` | int | 13 | Number of frames to sample per sequence |
| `--save_interval` | int | 50 | Save intermediate results every N sequences |

## Directory Structure

Expected directory structure of frames:
```
root_directory/
├── TR0/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
├── TR1/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
└── ...
```


## Guidelines

1. **Frame Sampling**: The tool evenly samples frames across each sequence to ensure consistent analysis.

2. **Majority Voting**: Results are determined by majority voting across sampled frames.


