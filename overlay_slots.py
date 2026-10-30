import argparse
import json
import cv2
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--slots", required=True)
    ap.add_argument("--frames", type=int, nargs="*", default=[0, 300])
    ap.add_argument("--save", default="overlay_slots.png")
    args = ap.parse_args()

    slots = json.load(open(args.slots, encoding="utf-8"))
    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    xs = [p[0] for s in slots for p in s["coordinates"]]
    ys = [p[1] for s in slots for p in s["coordinates"]]
    print(f"Video: {W}x{H}, {n} frame | {len(slots)} ô | Tọa độ x: {min(xs)}..{max(xs)}, y: {min(ys)}..{max(ys)}")

    if max(xs) > W or max(ys) > H:
        print("⚠️ CẢNH BÁO: Tọa độ ô nằm ngoài khung hình -> slots.json vẽ cho video khác resolution!")

    panels = []
    for f in args.frames:
        f = min(max(f, 0), n - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if not ok: continue
        for s in slots:
            cv2.polylines(im, [np.array(s["coordinates"], np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(im, f"frame {f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        panels.append(im)
    cap.release()
    cv2.imwrite(args.save, np.vstack(panels) if len(panels) > 1 else panels[0])
    print(f"✅ Đã lưu hình ảnh kiểm tra tại: {args.save}")

if __name__ == "__main__":
    main()
