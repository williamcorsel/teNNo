import shutil
import os
import sys
import random
import argparse


def main():
	parser = argparse.ArgumentParser(description="Preprocess folder from oi_download_dataset")
	parser.add_argument("--name", "-n", dest="NAME")
	parser.add_argument("--prefix", "-p", dest="PREFIX")
	args = parser.parse_args()

	path = args.PREFIX + args.NAME + "/"

	move_labels(path)
	split_data(path)

def move_labels(classes_path):
	print("Start copying files...")
	for class_ in os.listdir(classes_path):
		
		print("Copying class " + class_)
		base_folder = os.path.join(classes_path, class_)
				
		for sub in os.listdir(base_folder):	
			file_folder = os.path.join(base_folder, sub)
			if os.path.isdir(file_folder):
				for file_ in os.listdir(file_folder):
					shutil.move(os.path.join(file_folder, file_), os.path.join(base_folder, file_))
		
				os.rmdir(file_folder)
	print("Done copying files!")

def split_data(classes_path):
	print("Create train/test files")
	classes = os.listdir(classes_path)
	train = open(os.path.join(classes_path, "train.txt"), "w")
	test = open(os.path.join(classes_path, "test.txt"), "w")
	
	for class_ in classes:
		base_folder = os.path.join(classes_path, class_)
		
		for file_ in os.listdir(base_folder):
			if file_.endswith(".jpg"):
				file_path = os.path.join(base_folder, file_)
				if random.randint(1,100) <= 20:
					test.write(file_path + "\n")
				else:
					train.write(file_path + "\n")

	print("Done creating files!")

if __name__ == '__main__':
	main()
	

	
