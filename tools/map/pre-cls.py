import os


def file_name(file_dir):
    with open("val.txt", 'w') as f:
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                img_name = file.split(".")[0]
                f.write(img_name)
                f.write("\n")


def cls_pred_file(pred_file):
    with open(pred_file) as f:
        lines = f.readlines()
        classes_name = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair",
                        "cow", "diningtable", "dog", "horse",
                        "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train", "tvmonitor"]
        for cls in classes_name:
            with open("/home/zhangz/DL/zcnn/tools/map/pred_out/%s.txt" % cls, 'w') as F:
                print("Writing %s.txt" % cls)
                for line in lines:
                    img_name = line.strip().split(" ")[0]
                    objects = line.strip().split(" ")[1:]
                    for i in range(len(objects)):
                        score = objects[i].split(",")[0]
                        x1 = objects[i].split(",")[1]
                        y1 = objects[i].split(",")[2]
                        x2 = objects[i].split(",")[3]
                        y2 = objects[i].split(",")[4]
                        label = int(objects[i].split(",")[5])
                        if classes_name[label] == cls:
                            F.write(img_name + " " + score + " " +
                                    x1 + " " + y1 + " " + x2 + " " + y2)
                            F.write("\n")
            print("%s.txt is done!" % cls)


if __name__ == "__main__":
    # file_name("./datasets/score/labels/val")
    print("hello world!!")
    cls_pred_file("/home/zhangz/DL/zcnn/tools/map/out.txt")
