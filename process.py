import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
from Net import Net


parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", required=True, help="path or .h5 file with the model")
parser.add_argument("-f", "--file", help="image file for procssing")
parser.add_argument("-d", "--directory", help="path to the folder for processing intire files")



args = parser.parse_args()

if not (args.file or args.directory):
    print("file or directory required")
    exit(322)

net = Net(args.model)

if args.file:
    net.process_file(args.file)

if args.directory:
    net.process_directory(args.directory)



