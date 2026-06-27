#!/usr/bin/env python3
"""Export a trained YOLO26 drone model for Magic Leap 2 (ONNX FP32 / FP16 / INT8).

ML2 path: ONNX -> ONNX Runtime (+MLSDK C API), CPU backend XNNPACK. The NMS-free one-to-one
head is kept (we do NOT pass end2end/nms), so the model emits (1, 300, 6) =
[x1, y1, x2, y2, score, class] and the device needs no NMS.

Artifacts (in --outdir, default weights/):
    <stem>_fp32.onnx   ultralytics export: opset17, dynamic=False, simplify, batch1, imgsz640
    <stem>_fp16.onnx   onnxconverter-common float16 conversion of the FP32 graph
    <stem>_int8.onnx   onnxruntime static PTQ (QDQ), calibrated on val images

INT8 affine mapping: for a real value r, scale s, zero-point z:  q = round(r / s) + z.
Calibration preprocessing matches inference: letterbox to imgsz, /255, RGB, CHW, float32.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/yolo26n_drone_640.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--outdir", default="weights")
    ap.add_argument("--calib-dir", default="/mnt/ssd_0/dataset/dut_yolo/images/val")
    ap.add_argument("--calib-num", type=int, default=200, help="100-300 recommended")
    ap.add_argument("--stem", default=None, help="output basename (default from weights)")
    ap.add_argument("--skip-int8", action="store_true")
    ap.add_argument("--skip-fp16", action="store_true")
    return ap.parse_args()


def letterbox(img, new=640, color=114):
    """Resize keeping aspect ratio, pad to a square `new`x`new`. Returns CHW float32 [0,1]."""
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, 3), color, dtype=np.uint8)
    top, left = (new - nh) // 2, (new - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None]  # (1,3,new,new)


def export_fp32(weights, imgsz, opset, dst):
    model = YOLO(weights)
    # NMS-free head kept (no end2end/nms args). Static batch=1, fixed imgsz, simplified.
    # NOTE: do not pass device="cpu" here — ultralytics then sets CUDA_VISIBLE_DEVICES=-1
    # for the whole process, which forces the FP16 step onto the buggy converter fallback.
    onnx_path = model.export(format="onnx", opset=opset, dynamic=False,
                             simplify=True, batch=1, imgsz=imgsz)
    shutil.copy2(onnx_path, dst)
    print(f"[FP32] {dst}")
    return dst


def export_fp16(weights, fp32_path, dst, imgsz, opset):
    """Primary: ultralytics native half=True (needs CUDA). Fallback: onnxconverter-common
    float16 with Resize blocked (keep_io_types alone yields an invalid cast around Resize)."""
    import torch
    if torch.cuda.is_available():
        try:
            p = YOLO(weights).export(format="onnx", half=True, opset=opset, dynamic=False,
                                     simplify=True, batch=1, imgsz=imgsz, device=0)
            shutil.copy2(p, dst)
            print(f"[FP16] {dst}  (native half)")
            return dst
        except Exception as e:  # noqa: BLE001
            print(f"[FP16] native half failed ({e}); using float16 converter")
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(str(fp32_path))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True, op_block_list=["Resize"])
    onnx.save(m16, str(dst))
    print(f"[FP16] {dst}  (float16 converter)")
    return dst


class CalibReader:
    """onnxruntime CalibrationDataReader over a sample of val images."""
    def __init__(self, calib_dir, n, imgsz, input_name):
        files = sorted(p for p in Path(calib_dir).iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
        # Evenly sample n files across the split for a representative distribution.
        if len(files) > n:
            idx = np.linspace(0, len(files) - 1, n).astype(int)
            files = [files[i] for i in idx]
        self.files = files
        self.imgsz = imgsz
        self.input_name = input_name
        self._it = iter(self.files)

    def get_next(self):
        f = next(self._it, None)
        if f is None:
            return None
        img = cv2.imread(str(f))
        return {self.input_name: letterbox(img, self.imgsz)}

    def rewind(self):
        self._it = iter(self.files)


def export_int8(fp32_path, dst, calib_dir, n, imgsz):
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = Path(str(fp32_path).replace(".onnx", "_pre.onnx"))
    quant_pre_process(str(fp32_path), str(pre))  # shape inference + cleanup for QDQ

    input_name = ort.InferenceSession(
        str(pre), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    reader = CalibReader(calib_dir, n, imgsz, input_name)

    # Quantize ONLY Conv layers. The NMS-free detection head concatenates box coords
    # (0..640) with scores (0..1) in one tensor; per-tensor activation quantization there
    # crushes the tiny score values to 0 (model outputs all-zero scores). Keeping the head
    # and activations in float preserves accuracy while Conv INT8 gives most of the speedup.
    quantize_static(
        model_input=str(pre),
        model_output=str(dst),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,          # QDQ static quantization
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv"],
    )
    pre.unlink(missing_ok=True)
    print(f"[INT8] {dst}  (calibrated on {len(reader.files)} imgs, QDQ per-channel)")
    return dst


def main():
    a = parse_args()
    import torch
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    stem = a.stem or Path(a.weights).stem
    fp16_dst = out / f"{stem}_fp16.onnx"

    # FP16 native (half=True) must run BEFORE the FP32 export: ultralytics' ONNX export
    # disables CUDA for the process, after which FP16 can only use the converter fallback.
    if not a.skip_fp16 and torch.cuda.is_available():
        export_fp16(a.weights, None, fp16_dst, a.imgsz, a.opset)

    fp32 = export_fp32(a.weights, a.imgsz, a.opset, out / f"{stem}_fp32.onnx")

    if not a.skip_fp16 and not fp16_dst.exists():   # no GPU -> converter fallback needs fp32
        export_fp16(a.weights, fp32, fp16_dst, a.imgsz, a.opset)
    if not a.skip_int8:
        export_int8(fp32, out / f"{stem}_int8.onnx", a.calib_dir, a.calib_num, a.imgsz)

    print("\n=== sizes ===")
    for p in sorted(out.glob(f"{stem}_*.onnx")):
        print(f"  {p.name:32} {p.stat().st_size/1e6:7.2f} MB")


if __name__ == "__main__":
    main()
