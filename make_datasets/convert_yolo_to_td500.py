import os
import re
from typing import List, Tuple


def yolo_to_box(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """YOLO normalized bbox to pixel box (left, top, right, bottom)."""
    x_c = cx * img_w
    y_c = cy * img_h
    bw = w * img_w
    bh = h * img_h
    left = max(0, min(img_w - 1, int(round(x_c - bw / 2.0))))
    top = max(0, min(img_h - 1, int(round(y_c - bh / 2.0))))
    right = max(0, min(img_w - 1, int(round(x_c + bw / 2.0))))
    bottom = max(0, min(img_h - 1, int(round(y_c + bh / 2.0))))
    return left, top, right, bottom


def box_to_quad(left: int, top: int, right: int, bottom: int) -> List[int]:
    """Axis-aligned rectangle to 4-point polygon (x1,y1,...,x4,y4)."""
    return [
        left, top,          # x1,y1 (top-left)
        right, top,         # x2,y2 (top-right)
        right, bottom,      # x3,y3 (bottom-right)
        left, bottom        # x4,y4 (bottom-left)
    ]


def main():
    gts_dir = r'datasets/dongsim_eng_gts'
    prefix = 'dongsim_eng_'
    img_ext = 'jpg'
    IMG_W, IMG_H = 1920, 1080
    rename_yolo = False


    # 경로 보정
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.isabs(gts_dir):
        gts_dir = os.path.join(project_root, gts_dir)

    if not os.path.isdir(gts_dir):
        raise SystemExit(f"Not found: {gts_dir}")

    img_ext = img_ext.lstrip('.').lower()
    if img_ext == 'jpeg':
        img_ext = 'jpg'

    # Any .txt; we'll extract the first digit-run from the base name
    pat_txt = re.compile(r'^(?P<base>.+)\.txt$', re.IGNORECASE)

    count_in, count_out = 0, 0
    for gt_name in sorted(os.listdir(gts_dir)):
        m = pat_txt.match(gt_name)
        if not m:
            continue
        base = m.group('base')  # filename without .txt
        mnum = re.search(r'\d+', base)
        if not mnum:
            # skip files without digits
            continue
        first_digit_pos = mnum.start()
        tail = base[first_digit_pos:]  # keep digits and any trailing postfix
        yolo_path = os.path.join(gts_dir, gt_name)

        # Target GT filename pairs with image name: <prefix><tail>.<img_ext>.txt
        out_base = f'{prefix}{tail}'
        out_name = f'{out_base}.{img_ext}.txt'
        out_path = os.path.join(gts_dir, out_name)

        with open(yolo_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        quads = []
        for ln in lines:
            # YOLO: <cls> <cx> <cy> <w> <h>
            parts = ln.split()
            if len(parts) != 5:
                # skip malformed
                continue
            try:
                _, cx, cy, w, h = parts
                cx = float(cx); cy = float(cy); w = float(w); h = float(h)
            except Exception:
                continue
            left, top, right, bottom = yolo_to_box(cx, cy, w, h, IMG_W, IMG_H)
            quad = box_to_quad(left, top, right, bottom)
            quads.append(quad)

        with open(out_path, 'w', encoding='utf-8') as f:
            for quad in quads:
                line = ','.join(str(v) for v in quad) + ',0'  # 0 = text (trainable)
                f.write(line + '\n')

        # Optionally rename original YOLO file to <prefix><tail>.txt
        if rename_yolo:
            new_yolo_name = f'{out_base}.txt'
            new_yolo_path = os.path.join(gts_dir, new_yolo_name)
            if os.path.abspath(new_yolo_path) != os.path.abspath(yolo_path):
                if os.path.exists(new_yolo_path):
                    raise SystemExit(f"Target YOLO file already exists: {new_yolo_path}")
                os.rename(yolo_path, new_yolo_path)

        count_in += 1
        count_out += len(quads)

    print(f"Converted {count_in} YOLO files -> wrote {count_out} TD500 lines in {gts_dir}.")


if __name__ == '__main__':
    main()


