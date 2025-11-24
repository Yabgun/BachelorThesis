import argparse
import sys
from scripts.fetch_kaggle_healthcare import main as kaggle_main
from scripts.augment_images_from_github import main as github_images_main
from scripts.extract_cxr_features import main as extract_cxr_main
from scripts.prepare_healthcare import main as prepare_healthcare_main
from scripts.build_multimodal_dataset import main as build_multimodal_main


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kaggle", action="store_true")
    p.add_argument("--github-images", action="store_true")
    p.add_argument("--max-images", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    if args.kaggle:
        kaggle_main()
    if args.github_images:
        github_images_main(max_files=args.max_images)
    extract_cxr_main()
    prepare_healthcare_main()
    build_multimodal_main()


if __name__ == "__main__":
    main()



