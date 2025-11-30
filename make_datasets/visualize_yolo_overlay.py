import os
import argparse
from typing import List, Tuple

import cv2


def yolo_to_box(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    """YOLO normalized bbox를 픽셀 좌표 (left, top, right, bottom)로 변환."""
    x_c = cx * img_w
    y_c = cy * img_h
    bw = w * img_w
    bh = h * img_h

    left = max(0, min(img_w - 1, int(round(x_c - bw / 2.0))))
    top = max(0, min(img_h - 1, int(round(y_c - bh / 2.0))))
    right = max(0, min(img_w - 1, int(round(x_c + bw / 2.0))))
    bottom = max(0, min(img_h - 1, int(round(y_c + bh / 2.0))))
    return left, top, right, bottom


def load_yolo_annotations(txt_path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    YOLO 라벨 파일을 읽어서 (class_id, cx, cy, w, h) 리스트로 반환.
    각 줄 포맷: <cls> <cx> <cy> <w> <h>  (0~1 정규화 좌표)
    """
    annos: List[Tuple[int, float, float, float, float]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 5:
                # YOLO 형식이 아니면 스킵
                continue
            try:
                cls = int(float(parts[0]))
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except Exception:
                # 파싱 실패 시 해당 줄 스킵
                continue
            annos.append((cls, cx, cy, w, h))
    return annos


def draw_yolo_boxes_on_image(
    img,
    annos: List[Tuple[int, float, float, float, float]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    class_names: List[str] = None,
):
    """
    YOLO annotation들을 이미지 위에 박스로 그려서 반환.
    color: BGR 색상, thickness: 박스 선 굵기.
    """
    img_h, img_w = img.shape[:2]

    for cls, cx, cy, w, h in annos:
        left, top, right, bottom = yolo_to_box(cx, cy, w, h, img_w, img_h)

        # 박스 그리기
        cv2.rectangle(img, (left, top), (right, bottom), color, thickness)

        # 클래스 라벨 텍스트 (선택)
        if class_names is not None and 0 <= cls < len(class_names):
            label = class_names[cls]
        else:
            label = str(cls)

        text = label
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 텍스트 배경 박스
        cv2.rectangle(
            img,
            (left, max(0, top - th - baseline)),
            (left + tw, top),
            color,
            -1,
        )
        # 텍스트 자체
        cv2.putText(
            img,
            text,
            (left, top - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return img


def load_class_names(names_path: str) -> List[str]:
    """각 줄에 하나씩 class name이 있는 .txt 파일을 읽어서 리스트로 반환."""
    names: List[str] = []
    with open(names_path, "r", encoding="utf-8") as f:
        for ln in f:
            name = ln.strip()
            if not name:
                continue
            names.append(name)
    return names


def main():
    parser = argparse.ArgumentParser(
        description="YOLO txt + 이미지에 박스를 overlay 해서 시각화"
    )
    parser.add_argument("--img", required=True, help="입력 이미지 경로")
    parser.add_argument("--txt", required=True, help="YOLO 라벨 txt 경로")
    parser.add_argument(
        "--output",
        default=None,
        help="결과 이미지를 저장할 경로 (기본: <이미지이름>_yolo_vis.<ext>)",
    )
    parser.add_argument(
        "--class_names",
        default=None,
        help="각 줄에 하나씩 클래스 이름이 적힌 txt 파일 경로 (옵션)",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="박스 선 굵기 (기본: 2)",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="창에 띄우지 않고 저장만 할 때 사용",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.img):
        raise SystemExit(f"이미지 파일을 찾을 수 없습니다: {args.img}")
    if not os.path.isfile(args.txt):
        raise SystemExit(f"YOLO txt 파일을 찾을 수 없습니다: {args.txt}")

    # 클래스 이름 로드(선택)
    class_names = None
    if args.class_names is not None:
        if not os.path.isfile(args.class_names):
            raise SystemExit(f"class_names 파일을 찾을 수 없습니다: {args.class_names}")
        class_names = load_class_names(args.class_names)

    # 이미지 로드
    img = cv2.imread(args.img)
    if img is None:
        raise SystemExit(f"이미지를 열 수 없습니다: {args.img}")

    # YOLO 어노테이션 로드
    annos = load_yolo_annotations(args.txt)
    if not annos:
        print(f"경고: YOLO 어노테이션이 없거나 형식이 맞지 않습니다: {args.txt}")

    # 박스 그리기
    vis_img = draw_yolo_boxes_on_image(
        img,
        annos,
        color=(0, 255, 0),
        thickness=args.thickness,
        class_names=class_names,
    )

    # 출력 경로 결정
    if args.output is None:
        img_dir, img_name = os.path.split(args.img)
        name, ext = os.path.splitext(img_name)
        out_name = f"{name}_yolo_vis{ext}"
        out_path = os.path.join(img_dir, out_name)
    else:
        out_path = args.output

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis_img)
    print(f"시각화 결과 저장: {out_path}")

    # 화면에 보여주기 (옵션)
    if not args.no_show:
        cv2.imshow("YOLO Overlay", vis_img)
        print("창을 닫으려면 아무 키나 누르세요.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


