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

# helper function
# given a new set of image_ids, create a corresponding set of images
# images are selectively copied from the original set into a new directory
def getimages(ids, dirname):
    try:
        parent = os.path.dirname(os.path.abspath(__file__))
        imgs = os.path.join(parent, "train_images")
        dest = os.path.join(parent, dirname)
        os.mkdir(dest)

        for id in ids:
            # img = "%s.jpg" % (id)
            # shutil.copy(os.path.join(imgs, img), dest)
            shutil.copy(os.path.join(imgs, id), dest)
        
    except Exception as e:
        print(e)
        print("Could not create image set (if the directory already exists, delete/rename it and try again)")

# create a minimum balanced set of training images
# TODO: shuffle the rows in train.csv so a different minimized dataset is produced every time?
def minimize():
    dist = countlabels()
    minimum = dist[min(dist)] # gets the value of the class with least amt of labels in the set

    with open("train.csv") as labels:
        reader = csv.reader(labels)
        header = next(reader)
        ids = []
        counts = {key: 0 for key in ['0','1','2','3','4']}
        for row in reader:
            # id = row[0][:-4] # remove '.jpg'
            id = row[0]
            label = row[1]

            if counts[label] < minimum:
                ids.append(id)
                counts[label] += 1

        getimages(ids, "min_balanced_train_imgs")
        return counts


# print(minimize())
