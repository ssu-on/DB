import os
import re


def main():
    # ===== 사용자 설정(코드 내 하드코딩) =====
    # 변경 대상 GT 폴더 (프로젝트 루트 기준 상대경로 또는 절대경로)
    gts_dir = r'datasets/Subtitles/train_gts_dongsim2'
    # 접두어 (예: 'dongsim')
    prefix = 'dongsim'
    # 미리보기 모드(False면 실제로 rename)
    dry_run = False
    # ======================================

    # 경로 보정
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.isabs(gts_dir):
        gts_dir = os.path.join(project_root, gts_dir)

    if not os.path.isdir(gts_dir):
        raise SystemExit(f"Not found: {gts_dir}")

    # 패턴: prefix + 숫자 + ('.jpg.txt' or '.txt')
    pat_jpg_txt = re.compile(rf'^{re.escape(prefix)}(\d+)(\.jpg\.txt)$', re.IGNORECASE)
    pat_txt = re.compile(rf'^{re.escape(prefix)}(\d+)(\.txt)$', re.IGNORECASE)

    renamed, skipped = 0, 0
    for fname in sorted(os.listdir(gts_dir)):
        src = os.path.join(gts_dir, fname)
        if not os.path.isfile(src):
            continue

        m = pat_jpg_txt.match(fname)
        if not m:
            m = pat_txt.match(fname)
        if not m:
            skipped += 1
            continue

        number, suffix = m.group(1), m.group(2)
        new_name = f"{prefix}_{number}{suffix}"
        dst = os.path.join(gts_dir, new_name)

        # 이미 원하는 형태면 스킵
        if os.path.abspath(src) == os.path.abspath(dst):
            skipped += 1
            continue

        if os.path.exists(dst):
            raise SystemExit(f"Target exists: {dst}")

        print(f"rename: {fname} -> {new_name}")
        if not dry_run:
            os.rename(src, dst)
        renamed += 1

    print(f"Done. renamed={renamed}, skipped={skipped}, dir={gts_dir}")


if __name__ == '__main__':
    main()


