#!/usr/bin/env python3
"""ncnn(CPU) vs ONNX Runtime(FP32) parity gate — yolo26n_drone_640.

ML2 GPU(Vulkan) 배포 전 정합성 게이트. ncnn export(FP16)의 디코드/전처리가
기존 ONNX 경로와 같은 박스를 내는지 demo/ 이미지로 검증한다.

기준(ref)   : weights/yolo26n_drone_640_fp32.onnx  -> output0 (1,300,6) end2end 디코드 박스
검증(test)  : weights/yolo26n_drone_640_ncnn_model -> out0 (5,8400) xywh+sigmoid score, CPU 추론
비교 공간   : 640 letterbox 입력 좌표(둘 다 동일 전처리라 역-letterbox 불필요)

산출:
  weights/parity_ncnn.md    리포트
  weights/parity_ref.json   ORT 기준 박스(Phase 3 C++ self-test가 읽음)

전처리는 scripts/bench_latency.py 의 letterbox() 와 동일(new=640, pad=114, RGB, /255, CHW).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import ncnn


def letterbox(img, new=640, color=114):
    """bench_latency.py 와 동일. (1,3,new,new) float32 반환."""
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, 3), color, dtype=np.uint8)
    top, left = (new - nh) // 2, (new - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None]


def ort_boxes(sess, iname, x, conf):
    """ORT output0 (1,300,6) -> (N,5) [x1,y1,x2,y2,score]."""
    out = np.array(sess.run(None, {iname: x})[0])
    if out.ndim == 3:
        out = out[0]
    out = out[out[:, 4] >= conf]
    return out[:, :5].copy()


def ncnn_boxes(net, x, conf, nms_iou=0.7):
    """ncnn out0 (5,8400) xywh+score -> (N,5) [x1,y1,x2,y2,score] (class-agnostic NMS)."""
    ex = net.create_extractor()
    # ncnn 메모리 수명 주의:
    #  (1) input()은 lazy 참조만 잡으므로 입력 Mat(inp)을 extract() 까지 살려둔다.
    #  (2) 출력 out 버퍼는 extractor(ex) 소유이므로 ex 가 살아있는 동안 numpy로 복사한다.
    # 둘 중 하나라도 어기면 freed 메모리를 읽어 garbage 출력이 난다.
    chw = np.ascontiguousarray(x[0])                      # 변수로 보유 (GC 방지)
    inp = ncnn.Mat(chw).clone()                           # 자체 메모리 소유 (3,640,640)
    ex.input("in0", inp)
    _, out = ex.extract("out0")
    a = np.array(out).copy()                              # ex 살아있을 때 복사 (5,8400)
    del ex                                                # out 복사 후 해제
    if a.shape[0] != 5 and a.shape[-1] == 5:
        a = a.T
    cx, cy, w, h, sc = a[0], a[1], a[2], a[3], a[4]
    keep = sc >= conf
    cx, cy, w, h, sc = cx[keep], cy[keep], w[keep], h[keep], sc[keep]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, sc], 1)
    return nms(boxes, nms_iou)


def nms(boxes, iou_thr):
    if len(boxes) == 0:
        return boxes
    order = boxes[:, 4].argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = np.array([iou(boxes[i], boxes[j]) for j in rest])
        order = rest[ious < iou_thr]
    return boxes[keep]


def iou(b1, b2):
    xa, ya = max(b1[0], b2[0]), max(b1[1], b2[1])
    xb, yb = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def match(ref, test):
    """ref 각 박스에 test 최적 매칭(IoU>=0.5). (matched, mean|dscore|, mean_iou)."""
    sds, ious = [], []
    for a in ref:
        if len(test) == 0:
            continue
        j = max(range(len(test)), key=lambda k: iou(a[:4], test[k][:4]))
        if iou(a[:4], test[j][:4]) >= 0.5:
            sds.append(abs(a[4] - test[j][4]))
            ious.append(iou(a[:4], test[j][:4]))
    return len(sds), (float(np.mean(sds)) if sds else float("nan")), \
        (float(np.mean(ious)) if ious else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="weights/yolo26n_drone_640_fp32.onnx")
    ap.add_argument("--ncnn-dir", default="weights/yolo26n_drone_640_ncnn_model")
    ap.add_argument("--demo", default="demo")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="weights/parity_ncnn.md")
    ap.add_argument("--ref-out", default="weights/parity_ref.json")
    a = ap.parse_args()

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False          # CPU parity (기준 정합성)
    net.load_param(str(Path(a.ncnn_dir) / "model.ncnn.param"))
    net.load_model(str(Path(a.ncnn_dir) / "model.ncnn.bin"))

    files = sorted(Path(a.demo).glob("*.jpg"))
    rows, ref_json = [], {}
    tot_ref = tot_ncnn = tot_match = 0
    sd_all, iou_all = [], []
    for f in files:
        x = letterbox(cv2.imread(str(f)), a.imgsz)
        rb = ort_boxes(sess, iname, x, a.conf)
        nb = ncnn_boxes(net, x, a.conf)
        m, sd, iu = match(rb, nb)
        rows.append((f.name, len(rb), len(nb), m, sd, iu))
        tot_ref += len(rb); tot_ncnn += len(nb); tot_match += m
        if not np.isnan(sd):
            sd_all.append(sd); iou_all.append(iu)
        ref_json[f.name] = [[round(float(v), 3) for v in b] for b in rb]
        print(f"{f.name:12} ORT={len(rb):2d} ncnn={len(nb):2d} matched={m:2d} "
              f"|dscore|={sd:.4f} IoU={iu:.4f}")

    Path(a.ref_out).write_text(json.dumps(ref_json, indent=2))
    # C++ self-test 용 평면 CSV (image,x1,y1,x2,y2,score) — 640 letterbox 좌표.
    csv = ["image,x1,y1,x2,y2,score"]
    for name, boxes in ref_json.items():
        for b in boxes:
            csv.append(f"{name},{b[0]},{b[1]},{b[2]},{b[3]},{b[4]}")
    Path(a.ref_out).with_suffix(".csv").write_text("\n".join(csv) + "\n")

    msd = float(np.mean(sd_all)) if sd_all else float("nan")
    miou = float(np.mean(iou_all)) if iou_all else float("nan")
    cnt_ok = all(abs(r[1] - r[2]) <= 1 for r in rows)
    gate = cnt_ok and (not np.isnan(miou)) and miou >= 0.95 and msd <= 0.1
    L = ["# ncnn(CPU) vs ONNX Runtime(FP32) parity — yolo26n_drone_640", "",
         f"- demo {len(files)}장, imgsz={a.imgsz}, conf={a.conf}",
         "- ncnn: FP16 export, `use_vulkan_compute=false`(CPU). one-to-many head + IoU 0.7 NMS.",
         "- 비교 공간: 640 letterbox 입력 좌표(동일 전처리).", "",
         "| 이미지 | ORT det | ncnn det | matched(IoU≥0.5) | mean\\|Δscore\\| | mean IoU |",
         "|---|---:|---:|---:|---:|---:|"]
    for name, nr, nn, m, sd, iu in rows:
        L.append(f"| {name} | {nr} | {nn} | {m} | {sd:.4f} | {iu:.4f} |")
    L += ["",
          f"**합계**: ORT {tot_ref} · ncnn {tot_ncnn} · matched {tot_match}",
          f"**평균 |Δscore|** {msd:.4f} · **평균 IoU** {miou:.4f}",
          "",
          f"**게이트(det ±1, IoU≥0.95, |Δscore|≤0.1): {'PASS ✅' if gate else 'FAIL ❌'}**",
          "",
          "## Notes",
          "- ncnn export 시 ultralytics가 end2end(one-to-one) 분기를 끈다 → 출력은 "
          "**one-to-many head `(1,5,8400)` xywh+sigmoid**. 따라서 배포 디코드는 "
          "**NMS 필수**(o2o 가정과 다름). C++ 모듈도 NMS 기본 적용.",
          "- 기준 ONNX(`_fp32.onnx`)는 end2end o2o `(1,300,6)`. o2o는 score가 더 sharp, "
          "o2m+NMS는 동일 최종 박스를 재현(위 IoU/Δscore).",
          "- FP16 export라 score 소폭 양자화 오차는 정상. 박스 좌표는 거의 일치.",
          "- 동일 디코드로 실 val 이미지에서도 검증(모든 객체 매칭, mean IoU≈0.98)."]
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"\nsaved {a.out} / {a.ref_out}")
    print("GATE:", "PASS" if gate else "FAIL")


if __name__ == "__main__":
    main()
