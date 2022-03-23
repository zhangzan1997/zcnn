import sys
import os
import random
import shutil


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1] == '.jpg':
            list_name.append(file_path)


if __name__ == '__main__':
    print("Usage : [options]")

    ImageSet_list_VOC2012 = "/home/zhangz/DL/DataSet/VOCdevkit/VOC2012/JPEGImages/"
    sample_path = "/home/zhangz/DL/DataSet/VOCsample/"
    imagefiles = []
    samplefiles = []
    listdir(ImageSet_list_VOC2012, imagefiles)

    if(os.path.isdir(sample_path) == False):
        os.mkdir(sample_path)
    else:
        shutil.rmtree(sample_path)
        os.mkdir(sample_path)

    for i in range(5000):
        j = random.randint(0, len(imagefiles)-1)
        samplefiles.append(imagefiles[j])
        del imagefiles[j]

    for samplefile in samplefiles:
        cmd = "cp {} {}".format(samplefile, sample_path)
        # print(cmd)
        os.system(cmd)
    print("sample images Done!")
