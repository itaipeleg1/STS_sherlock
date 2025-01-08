# Frame Sequence Analysis with LLaVA

A Python tool for analyzing sequences of image frames using the LLaVA (Large Language and Vision Assistant) model. This tool process sequences of frames to detect features
##
Versions required:
```
pip install transformers==4.46.3
pip install bitsandbytes==0.41.3
pip install accelerate==0.26.0
```

## Features

- Process sequences of image frames using LLaVA model
- Customizable frame sampling per sequence
- Support for different sequence ranges
- Automatic intermediate saving of results
- Configurable prompts for different types of analysis
- Command-line interface for easy use

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


