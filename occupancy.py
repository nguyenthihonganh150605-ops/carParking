"""Lõi logic ô đỗ (không phụ thuộc YOLO nên test được độc lập).

Quy trình cho mỗi ô, mỗi frame:
  1. raw    : max Occupancy Ratio > theta                      (chỉ dùng detector)
  2. track  : raw + điều kiện theo track_id (xem dưới)         (detector + ByteTrack)
  3. final  : majority voting trên `window` frame của `track`  (+ Temporal Filtering)

Điều kiện theo track_id (hysteresis):
  - Ô đang TRỐNG -> chỉ chuyển sang CÓ XE khi có MỘT track_id duy nhất
    nằm trong ô (ratio > theta) liên tục >= `dwell` frame VÀ gần như đứng yên.
    => xe chỉ chạy ngang qua ô không bị tính là đã đỗ.
  - Ô đang CÓ XE -> giữ trạng thái chừng nào còn xe đè ratio > theta,
    kể cả khi ByteTrack đổi ID sau che khuất (tránh nhấp nháy do ID switch).
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
    def __init__(self, slots, theta=0.5, window=9, dwell=15,
                 move_frac=0.15, prune_after=300):
        self.slots = slots
        self.theta = theta
        self.window = window
        self.dwell_min = dwell
        self.move_frac = move_frac          # độ dịch chuyển cho phép / kích thước xe
        self.prune_after = prune_after
        self.votes = {s["id"]: deque(maxlen=window) for s in slots}
        self.final = {s["id"]: False for s in slots}
        self.dwell = {}                      # (slot_id, track_id) -> số frame liên tiếp
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
                    new_dwell[key] = self.dwell.get(key, 0) + 1
                    cands.append((r, v))

            raw = max_ratio > self.theta
            owner = best_tid if raw else None

            if self.final[sid]:              # đang có xe -> giữ nếu còn xe đè
                track = raw
            else:                            # đang trống -> cần xe đỗ ổn định
                track = False
                for r, v in sorted(cands, key=lambda t: -t[0]):
                    if new_dwell[(sid, v["id"])] >= self.dwell_min and v["_still"]:
                        track, owner = True, v["id"]
                        break

            self.votes[sid].append(track)
            final = sum(self.votes[sid]) * 2 > len(self.votes[sid])
            self.final[sid] = final
            results.append({"slot_id": sid, "max_ratio": max_ratio, "raw": raw,
                            "track": track, "final": final,
                            "track_id": owner if final else None})

        self.dwell = new_dwell
        self.frame += 1
        if self.frame % self.prune_after == 0:
            old = [t for t, f in self.last_seen.items()
                   if self.frame - f > self.prune_after]
            for t in old:
                self.hist.pop(t, None)
                self.last_seen.pop(t, None)
        return results
