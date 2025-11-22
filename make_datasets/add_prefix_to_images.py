import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='이미지 파일명 앞에 prefix를 붙여서 이름 변경')
    parser.add_argument('--images_dir', default='datasets/nofilter', type=str, help='이미지가 있는 디렉토리 경로')
    parser.add_argument('--prefix', default='nofilter_', type=str, help='파일명 앞에 붙일 prefix')
    parser.add_argument('--convert_to_jpg', action='store_true', help='모든 이미지 확장자를 .jpg로 변경')
    parser.add_argument('--dry_run', action='store_true', help='실제로 변경하지 않고 미리보기만')
    args = parser.parse_args()

    # 경로 보정
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.isabs(args.images_dir):
        images_dir = os.path.join(project_root, args.images_dir)
    else:
        images_dir = args.images_dir

    if not os.path.isdir(images_dir):
        raise SystemExit(f"디렉토리를 찾을 수 없습니다: {images_dir}")

    # 이미지 확장자
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG'}
    files = [f for f in sorted(os.listdir(images_dir)) 
             if os.path.splitext(f)[1] in valid_exts]

    if not files:
        print(f"이미지 파일을 찾을 수 없습니다: {images_dir}")
        return

    renamed_count = 0
    skipped_count = 0

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}이미지 파일 이름 변경 시작...")
    print(f"디렉토리: {images_dir}")
    print(f"Prefix: {args.prefix}")
    if args.convert_to_jpg:
        print(f"확장자 변경: 모든 파일을 .jpg로 변경")
    print(f"총 {len(files)}개 파일\n")

    for fname in files:
        old_path = os.path.join(images_dir, fname)
        name, old_ext = os.path.splitext(fname)
        
        # 이미 prefix가 붙어있는지 확인
        if name.startswith(args.prefix):
            print(f"  건너뜀 (이미 prefix 있음): {fname}")
            skipped_count += 1
            continue

        # 확장자 처리
        if args.convert_to_jpg:
            new_ext = '.jpg'
        else:
            new_ext = old_ext.lower()
            if new_ext == '.jpeg':
                new_ext = '.jpg'

        # 새 파일명: prefix + 원본 파일명 (확장자 변경)
        new_fname = args.prefix + name + new_ext
        new_path = os.path.join(images_dir, new_fname)

        # 같은 이름이면 건너뜀 (prefix도 같고 확장자도 같은 경우)
        if os.path.abspath(old_path) == os.path.abspath(new_path):
            skipped_count += 1
            continue

        # 이미 존재하는 파일이면 에러
        if os.path.exists(new_path):
            print(f"  오류: 대상 파일이 이미 존재합니다: {new_fname}")
            continue

        if args.dry_run:
            print(f"  [변경 예정] {fname} -> {new_fname}")
        else:
            os.rename(old_path, new_path)
            print(f"  변경됨: {fname} -> {new_fname}")
        
        renamed_count += 1

    print(f"\n완료!")
    print(f"  변경된 파일: {renamed_count}개")
    print(f"  건너뛴 파일: {skipped_count}개")


if __name__ == '__main__':
    main()

