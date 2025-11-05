import os
import re


def main():
    images_dir = r'datasets/Subtitles/pamyo216'
    list_path = r'datasets/Subtitles/pamyo216.txt'
    prefix = 'pamyo216_'
    ext = 'jpg'
    match_regex = r'(\d+)'
    start_index = 1
    zero_pad = 0

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.isabs(images_dir):
        images_dir = os.path.join(project_root, images_dir)
    if not os.path.isabs(list_path):
        list_path = os.path.join(project_root, list_path)

    if not os.path.isdir(images_dir):
        raise SystemExit(f"Not found: {images_dir}")

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    files = [f for f in sorted(os.listdir(images_dir)) if os.path.splitext(f)[1].lower() in valid_exts]

    pattern = re.compile(match_regex, re.IGNORECASE)
    counter = start_index if start_index is not None else 1
    new_names = []

    for fname in files:
        old_path = os.path.join(images_dir, fname)
        name, old_ext = os.path.splitext(fname)
        m = pattern.search(fname)
        if m and m.groups():
            idx = m.group(1)
        else:
            idx = str(counter)
            counter += 1
        if zero_pad and idx.isdigit():
            idx = idx.zfill(zero_pad)
        new_ext = old_ext.lower().lstrip('.') if ext.lower() == 'keep' else ext.lstrip('.').lower()
        if new_ext in ['jpeg']:
            new_ext = 'jpg'
        new_fname = f"{prefix}{idx}.{new_ext}"
        new_path = os.path.join(images_dir, new_fname)

        if os.path.abspath(old_path) == os.path.abspath(new_path):
            new_names.append(new_fname)
            continue

        if os.path.exists(new_path):
            raise SystemExit(f"Target file already exists: {new_path}")

        os.rename(old_path, new_path)
        new_names.append(new_fname)

    os.makedirs(os.path.dirname(list_path), exist_ok=True)
    with open(list_path, 'w', encoding='utf-8') as f:
        for name in new_names:
            f.write(name + '\n')

    print(f"Renamed {len(new_names)} files in {images_dir} and wrote list: {list_path}")


if __name__ == '__main__':
    main()


