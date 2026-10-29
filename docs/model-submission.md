# Submitting a model to TANUH

You upload two things, plus one optional file:

1. **A model file** in one of three formats: ONNX, TorchScript or Hugging Face.
2. **`adaptor.py`**, which turns your model's raw outputs into the dataset's `predictions.csv`.
3. **`input_spec.json`** (optional, ONNX and TorchScript only): the model's input size and resize method. A TorchScript model needs it unless the size is embedded in the `.pt`; see below.

The platform runs your model exactly as you uploaded it. **It does not convert, retrain or fix your model.**
Before scoring, it does only these steps to every image:

1. Decode the file to RGB, detecting the type from its content, so file extensions don't matter:
   - **Images:** JPEG, PNG, TIFF, BMP, WebP and GIF.
   - **DICOM:** it takes the middle frame, inverts MONOCHROME1, min-max scales to 0–255 and copies the grey channel into 3 channels.
2. Resize to your model's input size.
3. Scale to 0–1 (float inputs) or keep 0–255 (uint8 inputs).
4. Arrange the channels and layout your model declares.

**Your model file must include every other preprocessing step your training used**, for example ImageNet mean/std, colour balancing or BGR order.
If a step is missing, nothing fails: your model simply scores lower. Test locally with `tools/check_model.py` before you upload.

## Choosing a format

| Your preprocessing | Use |
|---|---|
| Only the standard steps a Hugging Face image processor supports: resize, rescale, mean/std normalise, centre crop | Hugging Face, ONNX or TorchScript |
| Anything custom, such as colour balancing, BGR order or CLAHE | ONNX or TorchScript, with the steps built into the model graph |

## Hugging Face

Upload **one `.zip`** of the folder that `save_pretrained()` writes.

The platform loads it with:

```python
AutoImageProcessor.from_pretrained(dir, local_files_only=True, trust_remote_code=False)
AutoModelForImageClassification.from_pretrained(dir, local_files_only=True, use_safetensors=True,
                                                trust_remote_code=False, torch_dtype=torch.float32)
```

It runs offline with `transformers` 4.46.3 and `torch` 2.5.1. Your model gets **no internet access and runs none of your Python code**.

### How to create the zip

```python
model.save_pretrained("my-model", safe_serialization=True)   # config.json + model.safetensors
processor.save_pretrained("my-model")                         # preprocessor_config.json
```

Then zip the folder. The files can sit at the root of the zip or in any folder inside it. The platform uses the **shallowest folder that holds a `config.json`**, so a zip of a whole training run (`run/config.json` next to `run/checkpoint-500/config.json`) uses `run/`.

### What the platform uses from that folder

| File | What it must be |
|---|---|
| `config.json` | Valid JSON with a `model_type` that `transformers` 4.46.3 supports for image classification, such as `vit`, `convnext`, `resnet`, `mobilevitv2`, `swin`, `efficientnet`, `deit` or `beit`. |
| `*.safetensors` | The weights: `model.safetensors`, or shards plus `model.safetensors.index.json`. |
| `preprocessor_config.json` | Your image processor's settings: `size`, `do_resize`, `do_rescale`, `do_normalize`, `image_mean`, `image_std`, `resample`, `crop_size`. **This is your preprocessing.** The processor receives RGB `uint8` images straight from step 1 above; the platform does no resizing or scaling. |

### Ignored

Everything else in the zip is left out and never loaded, so it's safe to include:
- `README.md`, `generation_config.json` and `.gitattributes`
- Trainer files like `training_args.bin`, `optimizer.pt` and `rng_state.pth`
- `.py` files
- `__MACOSX/` folders and symlinks

If `config.json` or `preprocessor_config.json` contains `auto_map` (a pointer to custom model code), it's removed, and transformers' built-in class for the `model_type` is used.

### Rejected

The job fails with `ModelFormatError` (error code 2) when:

| What | Why |
|---|---|
| The weights exist only as `pytorch_model.bin`, `.pt`, `.pth`, `.ckpt`, `.h5` or `.msgpack` | These load through pickle and can run code. Save with `model.save_pretrained(dir, safe_serialization=True)`. |
| `config.json`, `preprocessor_config.json` or `*.safetensors` is missing from the model folder | The model can't load without them. |
| The architecture isn't built into `transformers` | Custom model code is never run. |
| Two model folders sit side by side at the same depth | Upload one model. |
| A needed file (e.g. the weights) is a symlink | Unsafe. |

### Size limits

- The upload limit is the one the submit page shows.
- Unzipped, the zip must be at most 8 GiB and at most 10,000 files.

### Output

The platform saves the model's `logits` as `[N, num_labels]`, under the name `logits`.
The platform does not use `id2label` in `config.json`. Your adaptor decides how each output column maps to the dataset's `class_names`.

## ONNX

- **Upload:** `model.onnx`, plus the optional external-data file (`.onnx.data`) in the weights slot.
- **Input:** one image input in NCHW or NHWC layout, with 1 or 3 channels, as float32, float16 or uint8. The batch size can be dynamic or fixed.
- **Height and width:** read from the model. If they are dynamic, give them in the input spec (below).
- **Resize method:** `bilinear` by default. Set `bicubic` in the input spec, or as the model metadata key `tanuh.resize`.
- **Your preprocessing:** add it as graph nodes in front of your network.

## TorchScript

- **Upload:** a `.pt` saved with `torch.jit.save`. Files saved with `torch.save` (state dicts or pickles) are rejected.
- **Input size:** TorchScript files don't record it, so give it once, either way:
  - **on the submit form** (uploaded as `input_spec.json`, below), or
  - **embedded in the file**:

    ```python
    torch.jit.save(scripted, "model.pt", _extra_files={"tanuh.json": '{"input_size": [256, 256], "resize": "bicubic"}'})
    ```

    Both `tanuh.json` and `input_spec.json` work as the embedded file's name. If you do both, the form wins.
- **Defaults:** everything except the size is optional: 3 channels, NCHW, float32, bilinear.
- **Your preprocessing:** put it inside the scripted module's `forward`.
- **Loading:** the model is loaded with `torch` 2.5.1.

## input_spec.json (ONNX and TorchScript, optional)

A small JSON file that describes the model's input. The key names are flexible:

| Key (any of) | Value | Example |
|---|---|---|
| `input_size`, `input_shape`, `image_size`, `shape`, `size` | an integer, `[H, W]`, `[C, H, W]`, `[N, C, H, W]`, `[H, W, C]` or `{"height", "width"}` | `224`, `[256, 256]`, `[3, 224, 224]` |
| `height`, `width`, `channels` | integers | `"channels": 1` for a grey model |
| `layout`, `data_format` | `NCHW` / `channels_first`, or `NHWC` / `channels_last` | `"layout": "NHWC"` |
| `dtype` | `float32`, `float16`, `uint8` | `"dtype": "float16"` |
| `resize`, `interpolation`, `resample` | `bilinear` / `linear` / `2`, or `bicubic` / `cubic` / `3` | `"interpolation": "cubic"` |

**Refused keys:** `mean`, `std`, `normalize` and `channel_order` are refused rather than ignored. The platform doesn't apply those steps, so put them inside the model.

**ONNX conflicts:** for ONNX, a value that contradicts the model's own fixed input fails the job with a clear message.

## adaptor.py

It runs as:

```
python3 adaptor.py --raw-dir raw/ --spec dataset_spec.json --output-dir predictions/
```

It reads these inputs:
- **`raw/raw_outputs.npz`:** `ids` holds each sample's path in the dataset (e.g. `Suspicious/a.jpg`), and there is one array per model output.
- **`dataset_spec.json`:** holds `task_type`, `class_names` and `num_classes`.

It writes a table to `predictions/`. Name it `predictions.csv`, or any name if it's the only `.csv`/`.tsv` there. The standard columns are:

| Problem type | Columns | Template |
|---|---|---|
| `binary_classification` | `file,label,score`, where `score` is the probability of class 1 | `adaptor_sigmoid_binary.py` |
| `multiclass_classification` | `file,label,prob_0,…,prob_{C-1}`, where the probabilities sum to 1 | `adaptor_softmax_multiclass.py` |
| `segmentation` | `file,mask`: one class-index mask per image (0 = background), written as a lossless image under `predictions/`; `mask` is its path relative to `predictions/`. Masks may be at the model's output size; the evaluator resizes them to the ground truth (nearest neighbour). | `adaptor_argmax_segmentation.py` |
| `bounding_box` (object detection) | one row **per box**: `file,x_min,y_min,x_max,y_max,score,class`, in the **original image's pixels**; `class` is an index into `class_names` (no background class). An image with no detections needs no row. | `adaptor_boxes_detection.py` (**always adapt it**: detector outputs vary) |

**Detection runs one image at a time**, so the model must accept batch size 1 (or a dynamic batch). The platform resizes each image to the model's input size **without letterboxing**. `raw_outputs.npz` also holds `orig_hw` (each image's original height and width) and `input_hw` (the model's), which your adaptor uses to scale boxes back. Each model output is stored per image as `<name>`, `<name>__shape` and `<name>__offset`; the template's `per_image()` reads them.

The evaluator is lenient about the details:

| | Accepted |
|---|---|
| Delimiter | `,` tab `;` `\|` |
| File column | `file`, `filename`, `image`, `id`, `path`, …, else the first column. The id can be the full id, the file name, or the name without extension. |
| Label | `label`, `pred`, `prediction`, `predicted_class`, …, as an index or a class name. **Optional:** without it, the label is `score ≥ 0.5` (binary) or the most likely class. |
| Binary score | `score`, `prob`, `prob_1`, `probability`, `confidence`, `prob_<class 1 name>`, `logit`, … Values outside [0,1] are treated as logits. **Optional:** without it, AUC is not reported. |
| Multiclass probabilities | one column per class named the same way (`prob_0…`, `p0…`, `logit_0…`, `prob_<name>…`, or just the class names), or one `probs` column holding a list. Logits get a softmax; rows not summing to 1 are re-normalised. |
| Extra rows / columns | rows for files the ground truth doesn't list, and extra columns, are ignored |

**Still required:** every file must have exactly one prediction, so every model is scored on the same samples. For templates, see `tools/reference/adaptor_sigmoid_binary.py` and `adaptor_softmax_multiclass.py`.
