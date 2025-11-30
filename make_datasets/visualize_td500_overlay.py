import os
import argparse
from typing import List, Tuple

import cv2
import numpy as np


def parse_td500_line(line: str) -> Tuple[List[int], int]:
    """
    TD500/ICDAR 스타일 한 줄 파싱.
    형식 예: x1,y1,x2,y2,x3,y3,x4,y4,0
    반환: (좌표 8개 리스트 [x1,y1,...,x4,y4], label)
    """
    parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
    if len(parts) < 9:
        raise ValueError(f"Expected at least 9 values, got {len(parts)}: {line!r}")

    # 앞의 8개는 좌표
    coords = []
    for v in parts[:8]:
        coords.append(int(round(float(v))))

    # 9번째는 label (무시해도 되지만 그대로 반환)
    try:
        label = int(float(parts[8]))
    except Exception:
        label = 0

    return coords, label


def load_td500_annotations(txt_path: str) -> List[Tuple[List[int], int]]:
    """
    TD500/ICDAR 포맷 txt 파일을 읽어서
    (좌표 8개 리스트, label) 의 리스트를 반환.
    """
    annos: List[Tuple[List[int], int]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                coords, label = parse_td500_line(ln)
            except Exception:
                # 형식이 안 맞으면 스킵
                continue
            annos.append((coords, label))
    return annos


def draw_td500_boxes_on_image(
    img,
    annos: List[Tuple[List[int], int]],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
):
    """
    TD500/ICDAR 어노테이션들을 이미지 위에 폴리곤(4점)으로 그리기.
    color: BGR, thickness: 선 굵기.
    """
    h, w = img.shape[:2]

    for coords, label in annos:
        if len(coords) != 8:
            continue
        x1, y1, x2, y2, x3, y3, x4, y4 = coords

        # 이미지 범위 안으로 클램프
        pts = np.array(
            [
                [
                    max(0, min(w - 1, x1)),
                    max(0, min(h - 1, y1)),
                ],
                [
                    max(0, min(w - 1, x2)),
                    max(0, min(h - 1, y2)),
                ],
                [
                    max(0, min(w - 1, x3)),
                    max(0, min(h - 1, y3)),
                ],
                [
                    max(0, min(w - 1, x4)),
                    max(0, min(h - 1, y4)),
                ],
            ],
            dtype=np.int32,
        )

        pts = pts.reshape((-1, 1, 2))

        # 폴리곤 그리기
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

        # label 텍스트가 필요하면 왼쪽 위에 작게 표시 (0이면 생략해도 됨)
        text = str(label)
        if text:
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            x_text = pts[0, 0, 0]
            y_text = pts[0, 0, 1]

            cv2.rectangle(
                img,
                (x_text, max(0, y_text - th - baseline)),
                (x_text + tw, y_text),
                color,
                -1,
            )
            cv2.putText(
                img,
                text,
                (x_text, y_text - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return img


def main():
    parser = argparse.ArgumentParser(
        description="TD500/ICDAR 포맷 txt + 이미지에 폴리곤을 overlay 해서 시각화"
    )
    parser.add_argument("--img", required=True, help="입력 이미지 경로")
    parser.add_argument("--txt", required=True, help="TD500/ICDAR GT txt 경로")
    parser.add_argument(
        "--output",
        default=None,
        help="결과 이미지를 저장할 경로 (기본: <이미지이름>_td500_vis.<ext>)",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="폴리곤 선 굵기 (기본: 2)",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="창에 띄우지 않고 저장만 할 때 사용 (서버 환경 권장)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.img):
        raise SystemExit(f"이미지 파일을 찾을 수 없습니다: {args.img}")
    if not os.path.isfile(args.txt):
        raise SystemExit(f"GT txt 파일을 찾을 수 없습니다: {args.txt}")

    img = cv2.imread(args.img)
    if img is None:
        raise SystemExit(f"이미지를 열 수 없습니다: {args.img}")

    annos = load_td500_annotations(args.txt)
    if not annos:
        print(f"경고: TD500 어노테이션이 없거나 형식이 맞지 않습니다: {args.txt}")

    vis_img = draw_td500_boxes_on_image(
        img,
        annos,
        color=(0, 0, 255),
        thickness=args.thickness,
    )

    # 출력 경로 결정
    if args.output is None:
        img_dir, img_name = os.path.split(args.img)
        name, ext = os.path.splitext(img_name)
        out_name = f"{name}_td500_vis{ext}"
        out_path = os.path.join(img_dir, out_name)
    else:
        out_path = args.output

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_path, vis_img)
    print(f"시각화 결과 저장: {out_path}")

    # 화면에 보여주기 (옵션)
    if not args.no_show:
        cv2.imshow("TD500 Overlay", vis_img)
        print("창을 닫으려면 아무 키나 누르세요.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


