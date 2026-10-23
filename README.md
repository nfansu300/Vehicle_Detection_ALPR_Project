# Vehicle Detection and Automatic License Plate Recognition

This project explores a two-stage automatic license plate recognition (ALPR)
workflow for Moroccan plates. It includes object-detection training notebooks,
ONNX export and optimization experiments, model artifacts, and a CPU inference
pipeline intended to run on a Raspberry Pi 4.

## What the project does

The inference pipeline in
[`src/full_pipeline/full_pipeline.py`](./src/full_pipeline/full_pipeline.py)
processes images in two stages:

1. **Vehicle and plate detection** — runs an ONNX detector for buses, cars,
   license plates, motorcycles, and trucks.
2. **Plate text recognition** — crops detected plates and runs a character
   detector. Recognized characters are ordered across the plate and can be
   rendered using the configured Arabic character mapping.

The pipeline includes image letterboxing, output decoding and filtering, plate
cropping, optional annotated images and plate crops, and per-stage/end-to-end
latency summaries. Results are written as JSON files.

The notebooks document experiments for YOLOv8, YOLO11, and RT-DETR detection
training, YOLO11-based OCR, and ONNX FP32/FP16/INT8 export or optimization.
Notebook paths and dataset access are configured for Google Colab and must be
adapted to your own datasets and environment.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`src/full_pipeline/`](./src/full_pipeline/) | Two-stage ONNX inference pipeline |
| [`src/detection_training/`](./src/detection_training/) | YOLOv8, YOLO11, and RT-DETR training notebooks |
| [`src/detection_Optimization/`](./src/detection_Optimization/) | Detection ONNX export and optimization notebook |
| [`src/ocr_optimization/`](./src/ocr_optimization/) | OCR ONNX optimization notebook |
| [`models/detection/`](./models/detection/) | Detection model artifacts by framework and precision |
| [`models/ocr/`](./models/ocr/) | OCR model artifacts by framework and precision |
| [`models/best_selected_models/`](./models/best_selected_models/) | Selected ONNX model artifacts |
| [`experiments/`](./experiments/) | Training outputs and experiment artifacts |
| Pipeline output | Created under the directory passed to `--output-dir` |
| [`deployment/raspberry_pi_4/`](./deployment/raspberry_pi_4/) | Raspberry Pi deployment scaffold |
| [`docs/`](./docs/) | Project documentation |

Model artifacts are stored with Git LFS. Install Git LFS before cloning, or run
`git lfs pull` after cloning if model files appear as small text pointer files.

## Requirements

Use Python 3.11 or 3.12. The repository separates the lightweight CPU inference
dependencies from the notebook training and optimization dependencies because
the notebooks use PyTorch and GPU-enabled ONNX Runtime, while the Raspberry Pi
pipeline uses CPU ONNX Runtime and headless OpenCV.

### CPU inference and Raspberry Pi

Create a virtual environment and install the pinned runtime dependencies:

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Linux / Raspberry Pi OS**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Training and optimization notebooks

Use a separate environment on a training workstation or run the notebooks in
Google Colab. The notebook dependencies are pinned in
[`requirements-training.txt`](./requirements-training.txt):

```bash
python -m pip install -r requirements-training.txt
```

The training notebooks use CUDA. Install a PyTorch build compatible with your
GPU, CUDA, and operating system when the pinned PyTorch wheel is not appropriate
for your machine; update the PyTorch and torchvision pins to match. The ONNX
optimization notebooks specifically install `onnxruntime-gpu==1.26.0`; GPU
execution also requires a compatible CUDA/cuDNN runtime. Do not install CPU
and GPU ONNX Runtime packages into the same environment.

The notebooks import `google.colab` for Drive and notebook integration; that
module is supplied by Google Colab and is not part of the pip requirements.
Local users need to replace Colab-specific paths and data access with their
own setup.

## Running the inference pipeline

The pipeline accepts model and input paths on the command line. From the
repository root, for example:

```bash
python src/full_pipeline/full_pipeline.py \
  --det-model models/detection/onnx/fp16/yolov8n_100ep_fp16.onnx \
  --ocr-model models/ocr/onnx/int8/yolo11_ocr_int8.onnx \
  --images-dir path/to/input_images \
  --output-dir results/end_to_end/run_01
```

The input directory is searched recursively for JPEG, PNG, and BMP images.
Use `python src/full_pipeline/full_pipeline.py --help` to see all available
options. The defaults in the script's `Config` class are Raspberry Pi-specific;
pass explicit model and image paths or update that configuration for your
machine.

The output directory contains:

- `metrics_summary.json` — aggregate inference timings and run summary
- `per_image_results.json` — detections, recognized plate text, and timings
- `errors.json` — created only when images fail during processing
- `annotated/` — annotated images, unless `--no-annotate` is used
- `plate_crops/` — saved plate crops, unless `--no-crops` is used

## Model and data notes

The pipeline expects a detection model with the configured five-class mapping
and an OCR model whose output matches the decoder in the pipeline. If you use
different models, check the class order, tensor shapes, image sizes, and
confidence thresholds before inference. The `Config` class contains the
current model paths and these settings.

Training notebooks download or reference datasets through Roboflow and
Google Drive. Configure your own dataset workspace, project, version, and
storage paths before running them; the notebooks are experimental workflows,
not a turnkey dataset download.

## Credentials

Some training notebooks currently contain hard-coded Roboflow API credentials.
Revoke and rotate those credentials before using or sharing the notebooks.
Store replacement keys outside the repository, such as in environment
variables or your notebook platform's secret manager; never commit API keys.

## Limitations

- The pipeline is configured for a specific detector/OCR model pair and
  Moroccan plate character mapping; it is not a general-purpose ALPR package.
- Accuracy, latency, and Raspberry Pi performance depend on the models,
  hardware, input images, and runtime build. Benchmark on your target device.
- Dataset access, training outputs, and model configuration may need to be
  updated to reproduce an experiment.

## License

Choose a project license and replace the placeholder in
[`LICENSE`](./LICENSE) before
redistributing this project or its artifacts.
