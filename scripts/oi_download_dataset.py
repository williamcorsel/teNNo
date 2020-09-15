import csv
import argparse

from openimages.download import download_dataset

def main():
    parser = argparse.ArgumentParser(description="Download classes from OpenImages Dataset")
    parser.add_argument("--classfile", "-c", dest="CLASSFILE")
    parser.add_argument("--output", "-o", dest="OUTPUT")
    parser.add_argument("--limit", "-l", dest="LIMIT", default=70000)
    args = parser.parse_args()

    print("Reading class file")
    with open(args.CLASSFILE, newline='') as f:
        reader = csv.reader(f)
        classes = [row[0] for row in reader]

    print("Starting download")
    download_dataset(args.OUTPUT, classes, limit=args.LIMIT, annotation_format="darknet")


if __name__ == '__main__':
    main()