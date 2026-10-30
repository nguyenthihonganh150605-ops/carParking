"""Lõi logic ô đỗ (không phụ thuộc YOLO nên test được độc lập).

Quy trình cho mỗi ô, mỗi frame:
  1. raw    : max Occupancy Ratio > theta                      (chỉ dùng detector)
  2. track  : raw + điều kiện theo track_id (xem dưới)         (detector + ByteTrack)
  3. final  : trạng thái cuối, BẤT ĐỐI XỨNG (hysteresis):
        - trống -> có xe : majority voting `window` frame của `track`
        - có xe -> trống : chỉ khi MẤT xe liên tục >= `leave_frames` frame

Các sửa đổi so với bản cũ (nguyên nhân ô có xe vẫn xanh):
  [1] dwell không còn bị reset về 0 khi detector rớt 1 frame -> chỉ giảm dần (`grace`).
  [2] Ô có xe không còn bị đổi sang trống chỉ vì mất xe 3/5 frame
      -> phải mất liên tục `leave_frames` (~1.5 s) mới trả về trống.
  [3] Warm-up: xe ĐÃ đỗ sẵn từ frame đầu được nhận ngay, không phải chờ dwell.
  [4] Bỏ print debug (làm chậm), tính occupancy_ratio 1 lần/frame thay vì 2.
"""
from collections import deque, defaultdict
from shapely.geometry import Polygon, box as shp_box


def occupancy_ratio(slot_poly, vehicle_poly):
    """Area(Slot ∩ Vehicle) / Area(Slot)  -- KHÔNG phải IoU."""
    if not slot_poly.intersects(vehicle_poly):
        return 0.0
    return slot_poly.intersection(vehicle_poly).area / slot_poly.area


def make_slot(slot_id, coordinates):
    poly = Polygon(coordinates)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return {"id": slot_id, "coordinates": coordinates, "poly": poly}


class SlotEngine:
    def __init__(self, slots, theta=0.3, window=5, dwell=5,
                 move_frac=0.15, prune_after=300,
                 leave_frames=45, grace=1, warmup=None, require_still=False):
        """
        leave_frames : số frame LIÊN TIẾP không thấy xe mới đổi ô sang trống (30fps -> 45 ≈ 1.5s)
        grace        : mỗi frame detector rớt, dwell chỉ giảm `grace` thay vì về 0
        warmup       : số frame đầu video dùng raw trực tiếp (mặc định = 2*dwell)
        require_still: True -> chỉ tính 'đã đỗ' khi xe gần như đứng yên (bản cũ tính _still nhưng KHÔNG dùng)
        """
        self.slots = slots
        self.theta = theta
        self.window = window
        self.dwell_min = dwell
        self.move_frac = move_frac
        self.prune_after = prune_after
        self.leave_frames = leave_frames
        self.grace = grace
        self.warmup = 2 * dwell if warmup is None else warmup
        self.require_still = require_still
        self.votes = {s["id"]: deque(maxlen=window) for s in slots}
        self.final = {s["id"]: False for s in slots}
        self.miss = {s["id"]: 0 for s in slots}       # số frame liên tiếp mất xe
        self.dwell = {}                               # (slot_id, track_id) -> điểm dwell
        self.hist = defaultdict(lambda: deque(maxlen=dwell))
        self.last_seen = {}
        self.frame = 0

    def _is_still(self, tid, size):
        h = self.hist[tid]
        if len(h) < h.maxlen:
            return False
        xs = [p[0] for p in h]
        ys = [p[1] for p in h]
        spread = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        return spread < self.move_frac * size

    def update(self, vehicles):
        """vehicles: list[{"id": int, "box": (x1,y1,x2,y2)}] của frame hiện tại."""
        for v in vehicles:
            x1, y1, x2, y2 = v["box"]
            tid = v["id"]
            self.hist[tid].append(((x1 + x2) / 2, (y1 + y2) / 2))
            self.last_seen[tid] = self.frame
            v["_poly"] = shp_box(x1, y1, x2, y2)
            v["_size"] = max(((x2 - x1) * (y2 - y1)) ** 0.5, 1e-6)
        for v in vehicles:
            v["_still"] = self._is_still(v["id"], v["_size"])

        in_warmup = self.frame < self.warmup
        active_keys = set()
        new_dwell = {}
        results = []

        for s in self.slots:
            sid = s["id"]
            max_ratio, best_tid = 0.0, None
            cands = []                       # (ratio, vehicle) có ratio > theta
            for v in vehicles:
                r = occupancy_ratio(s["poly"], v["_poly"])
                if r > max_ratio:
                    max_ratio, best_tid = r, v["id"]
                if r > self.theta:
                    key = (sid, v["id"])
                    active_keys.add(key)

                    new_dwell[key] = self.dwell.get(key, 0) + 1

                    if sid in [75, 78]:
                        print(
                            f"frame={self.frame} slot={sid} "
                            f"id={v['id']} ratio={r:.3f} "
                            f"dwell={new_dwell[key]}"
                        )

                    cands.append((r, v))

            raw = max_ratio > self.theta
            owner = best_tid if raw else None
            was_final = self.final[sid]

            # ---- track: điều kiện vào ô ----
            if was_final:                    # đang có xe -> giữ nếu còn xe đè
                track = raw
            elif in_warmup:                  # [3] xe đỗ sẵn từ đầu video
                track = raw
            else:                            # đang trống -> cần xe đỗ ổn định
                track = False
                for r, v in sorted(cands, key=lambda t: -t[0]):
                    if new_dwell[(sid, v["id"])] >= self.dwell_min:
                        track, owner = True, v["id"]
                        break

            self.votes[sid].append(track)

            # ---- final: hysteresis bất đối xứng ----
            if was_final:                    # [2] chỉ rời ô khi mất xe đủ lâu
                self.miss[sid] = 0 if raw else self.miss[sid] + 1
                final = self.miss[sid] < self.leave_frames
            else:
                self.miss[sid] = 0
                final = sum(self.votes[sid]) * 2 > len(self.votes[sid])
                if in_warmup:
                    final = raw

            self.final[sid] = final
            results.append({"slot_id": sid, "max_ratio": max_ratio, "raw": raw,
                            "track": track, "final": final,
                            "track_id": owner if final else None})

        # [1] dwell: giữ lại key đang active; key vừa rớt thì chỉ giảm dần
        for key, val in self.dwell.items():
            if key not in active_keys and val - self.grace > 0:
                new_dwell[key] = val - self.grace
        self.dwell = new_dwell

        self.frame += 1
        if self.frame % self.prune_after == 0:
            old = [t for t, f in self.last_seen.items()
                   if self.frame - f > self.prune_after]
            for t in old:
                self.hist.pop(t, None)
                self.last_seen.pop(t, None)
            self.dwell = {k: v for k, v in self.dwell.items() if k[1] not in old}
        return results