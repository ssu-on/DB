import os
import re
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gts_subdir', default='train_gts_dongsim2', help='GT subdir under datasets/Subtitles')
    parser.add_argument('--output_list', default=None, help='Output list path; default datasets/Subtitles/train_list_dongsim2.txt')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    subtitles_root = os.path.join(project_root, 'datasets', 'Subtitles')
    gts_dir = os.path.join(subtitles_root, args.gts_subdir)

    if args.output_list is None:
        out_list = os.path.join(subtitles_root, 'train_list_dongsim2.txt')
    else:
        out_list = args.output_list

    if not os.path.isdir(gts_dir):
        raise SystemExit(f"Not found: {gts_dir}")

    # Match TD500-style files like dongsim_123.jpg.txt
    pat = re.compile(r'^(?P<img>.+\.(jpg|jpeg|png))\.txt$', re.IGNORECASE)

    names = []
    for fname in sorted(os.listdir(gts_dir)):
        m = pat.match(fname)
        if not m:
            continue
        img_name = m.group('img')
        names.append(img_name)

    if not names:
        print(f"No GT files matched in {gts_dir}")
    else:
        os.makedirs(os.path.dirname(out_list), exist_ok=True)
        with open(out_list, 'w', encoding='utf-8') as f:
            for n in names:
                f.write(n + '\n')
        print(f"Wrote {len(names)} names to {out_list}")


if __name__ == '__main__':
    main()


