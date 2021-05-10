import csv
import pandas as pd
import os
import shutil

# get class distribution from csv of image_id,label
def countlabels(csvfile):
    with open(csvfile) as labels:
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
            shutil.copy(os.path.join(imgs, id), dest)
        
    except Exception as e:
        print(e)
        print("Could not create image set (if the directory already exists, delete/rename it and try again)")

# given a list of (image_id, label), create a new csv
def getcsv(tups, fname):
    with open (fname, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow( ("image_id","label") )
        csvwriter.writerows(tups)

# given a csv of image_id,label, split into train-test-validation
# the remaining data (after train and test data are split) is used for validation
def splitcsv(csvfile, trainpct=0.6, testpct=0.2):
    with open(csvfile) as labels:
        tups = list(map(tuple, csv.reader(labels)))[1:] # gets tuples and removes header
        leng = len(tups)
        traindata = tups[ : int(leng*trainpct)]
        testdata = tups[len(traindata) : int(len(traindata) + leng*testpct)]
        valdata = tups[len(traindata) + len(testdata):] # the rest of the data is used as validation data
        # print("Train data: ", traindata)
        getcsv(traindata, "TRAIN_" + csvfile)
        print(("Distribution for " + "TRAIN_" + csvfile + ": "), countlabels("TRAIN_" + csvfile))
        # print("Test data: ", testdata)
        getcsv(testdata, "TEST_" + csvfile)
        print(("Distribution for " + "TEST_" + csvfile + ": "), countlabels("TEST_" + csvfile))
        # print("Validation data: ", valdata)
        getcsv(valdata, "VAL_" + csvfile)
        print(("Distribution for " + "VAL_" + csvfile + ": "), countlabels("VAL_" + csvfile))




# create a minimum balanced set of a given csv of training data
# TODO: shuffle the rows in the csv so a different minimized dataset is produced every time?
def minimize(traincsv):
    dist = countlabels(traincsv)
    minimum = dist[min(dist)] # gets the value of the class with least amt of labels in the set
    ids = []

    with open(traincsv) as labels:
        reader = csv.reader(labels)
        header = next(reader)
        counts = {key: 0 for key in ['0','1','2','3','4']}
        for row in reader:
            id = row[0]
            label = row[1]

            if counts[label] < minimum:
                ids.append( (id,label) )
                counts[label] += 1
    
    outfile = "min_balanced_" + traincsv
    getcsv(ids, outfile)
    print(("Distribution for " + outfile + ": "), countlabels(outfile))

splitcsv("train.csv") # get the splits
minimize("TRAIN_train.csv") # get minimum balance for the training split
