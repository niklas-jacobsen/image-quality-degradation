# Image Quality Degradation & Threshold Analysis Tool

This command-line application evaluates the robustness of object detection models against progressively degraded image quality. It applies various syntactic and semantic degradations to a dataset and identifies the quality threshold where detection performance significantly deteriorates.

## Requirements

- Python version 3.9 or newer
- Image folder containing the files to be analyzed

## Features

-   **Syntactic Degradations**: Resolution, Luminance, Contrast, Blur, Noise, Saturation.
-   **Semantic Degradations**: Motion Blur, Rain, Fog, Shadow, Snow, Chromatic Aberration.
-   **Flexible Configuration**: Supports CLI arguments, JSON configuration files, and sensible defaults.
-   **Automated Analysis**:
    *   Calculates **mAP (mean Average Precision)**, **Average Confidence**, and **Detection Count** at each level.
    *   Measures specific Image Quality (IQ) metrics: **Brightness**, **Contrast**, **Sharpness**, **Entropy**, **Overexposure**.
    *   Detects thresholds using Relative Drop, Safety Floor, and Kneedle (Elbow) algorithms.
-   **Dual Storage Modes**: Run entirely in RAM for speed or save generated images to disk for inspection.
-   **Validation Constraint Generation**: Produces a set of portable constraints (e.g., "Sharpness > 150") to filter real-world images.

## Installation

1.  **Clone the repository**
2.  **Set up a virtual environment**:
    ```bash
    python3 -m venv venv

    source venv/bin/activate  # Linux/macOS
    # or
    venv\Scripts\activate  # Windows
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Generate Thresholds

The main script `src/generate_thresholds.py` runs the degradation pipeline and calculates failure points.

**Basic Usage:**
```bash
python src/generate_thresholds.py --images-dir data/images --modifier blur
```

**Common Arguments:**

| Argument | Description |
| :--- | :--- |
| `--images-dir`, `-dir` | **Required**. Path to the folder containing input images. |
| `--modifier`, `-mod` | **Required**. The degradation modifier(s) to apply. Can be a single name (`blur`), a list (`['rain', 'fog']`), or a group (`syntactic`, `semantic`, `all`). |
| `--passes`, `-ps` | Number of degradation levels to simulate (default: 10). |
| `--step-percent`, `-st` | Percentage of quality reduction per pass (default: 10.0). |
| `--config`, `-c` | Path to a JSON configuration file to load arguments from. |
| `--output-dir`, `-out` | Directory to save results (default: `../outputs`). |
| `--storage-mode`, `-sm` | `ram` (default, faster) or `disk` (saves generated images). |
| `--plot`, `-pl` | If set, generates a performance plot. |

### Configuration File Example

For easy reusability, you can define all parameters in a JSON file (e.g., `config.json`):

```json
{
    "images_dir": "data/val_images",
    "output_dir": "experiments/batch_01",
    "modifier": ["rain", "snow", "fog"],
    "passes": 10,
    "step_percent": 10.0,
    "storage_mode": "ram"
}
```
Run with:
```bash
python src/generate_thresholds.py --config config.json
```

You may still override any of the arguments defined in the config file via the CLI.

### 2. Apply Thresholds (Validation)

Once you have generated a `thresholds_*.json` file, use `src/apply_thresholds.py` to validate real-world images against the derived constraints.

```bash
python src/apply_thresholds.py --images-dir data/real_world --thresholds-json ../outputs/thresholds_20231027_120000.json
```

**Arguments:**

| Argument | Description |
| :--- | :--- |
| `--images-dir`, `-dir` | **Required**. Path to real-world images. |
| `--thresholds-json`, `-json` | **Required**. Path to the JSON report from step 1. |
| `--output-json`, `-out` | Path to save the validation report (default: `validation_report.json`). |
| `--move-images-to`, `-move` | Optional path to move failed images to for inspection. |

## Supported Modifiers

### Syntactic Group
Technically driven degradations affecting pixel values directly.
*   **`resolution`**: Downsamples and upsamples to simulate lower resolution.
*   **`luminance`**: Reduces brightness.
*   **`contrast`**: Reduces contrast towards gray.
*   **`blur`**: Applies Gaussian blur.
*   **`noise`**: Adds Gaussian noise.
*   **`saturation`**: Desaturates color towards grayscale.

### Semantic Group
Context-aware degradations simulating environmental conditions (powered by Albumentations).
*   **`motion_blur`**: Simulates camera or object motion.
*   **`rain`**: Adds rain streaks.
*   **`fog`**: Adds fog overlays.
*   **`shadow`**: Casts random shadows.
*   **`snow`**: Adds snow particles.
*   **`chromatic_aberration`**: Simulates lens color fringing.

## Output

The `generate_thresholds.py` script produces a JSON report containing:
-   **Run Metadata**: Timestamp, configuration used.
-   **Analysis**: Detected thresholds and cutoff values for each modifier.
-   **Raw Scores**: mAP scores, average confidence, and detection counts for every pass.
-   **Metric Profiles**: Average brightness, contrast, sharpness, entropy, and overexposure at each level.

If `plot` is enabled, a performance curve image is also saved.

The `apply_thresholds.py` script produces a JSON report containing:
-   **Summary**: Total images, passed, and failed.
-   **Thresholds**: The thresholds used for validation.
-   **Details**: Per-image results including status, failure reasons, and metric profiles.

## Dependencies

-   [**Ultralytics YOLO**](https://github.com/ultralytics/ultralytics): For object detection inference.
-   [**Albumentations**](https://github.com/albumentations-team/albumentations): For advanced image augmentations (rain, snow, etc.).
-   [**OpenCV & NumPy**](https://github.com/opencv/opencv): For core image processing.
-   [**Matplotlib**](https://github.com/matplotlib/matplotlib): For plotting results.
