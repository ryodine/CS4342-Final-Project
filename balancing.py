import csv
import os
import shutil

# get class distribution from train.csv
def countlabels():
    with open("train.csv") as labels:
        reader = csv.reader(labels)
        header = next(reader)
        counts = {key: 0 for key in ['0','1','2','3','4']}

        for row in reader:
            counts[row[1]] += 1

        return counts

# given a new set of image_ids, create a corresponding set of images
# images are selectively copied from the original set into a new directory
def getimages(ids, dirname):
    try:
        parent = os.path.dirname(os.path.abspath(__file__))
        imgs = os.path.join(parent, "train_images")
        dest = os.path.join(parent, dirname)
        os.mkdir(dest)

        for id in ids:
            img = "%s.jpg" % (id)
            shutil.copy(os.path.join(imgs, img), dest)
        
    except Exception as e:
        print(e)
        print("Could not create image set (if the directory already exists, delete/rename it and try again)")

def minimize(): # TODO: make a minimum balanced set of training images
    pass

ids = [6103, 2272550, 690163]
getimages(ids, "TESTLOL")
