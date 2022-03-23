import math
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, cosine_similarity


def manhattan_distance(target_lines, base_lines):
    distance = 0
    for i in range(len(target_lines)):
        distance += abs(float(target_lines[i]) - float(base_lines[i]))

    print("manhattan_distance = {}".format(distance))
    print("manhattan_distance / n = {}".format(distance / len(target_lines)))
    return distance


def euclidean_distance(target_lines, base_lines):
    distance = 0
    for i in range(len(target_lines)):
        distance += abs(float(target_lines[i]) - float(base_lines[i])) ** 2

    distance = distance ** 0.5
    print("euclidean_distance = {}".format(distance))
    print("euclidean_distance / n = {}".format(distance / len(target_lines)))
    return distance


def cosine_distance(target_lines, base_lines):
    distance = 0
    AB = 0.0
    A = 0.0
    B = 0.0
    for i in range(len(target_lines)):
        AB += float(target_lines[i]) * float(base_lines[i])
        A += float(target_lines[i]) ** 2
        B += float(base_lines[i]) ** 2
    distance = AB / ((A ** 0.5) * (B ** 0.5))

    #d = cosine_similarity([target_lines, base_lines])
    #print("cosine_distance = {}".format(d))
    print("cosine_similarity = {}".format(distance))
    return distance


if __name__ == '__main__':
    modelname = "Mobilenet"

    target_file = open(
        "../models/{}/{}_featuremap.txt".format(modelname, modelname), 'r')
    base_file = open(
        "../models/{}/{}_int8_featuremap.txt".format(modelname, modelname), 'r')

    target_lines = target_file.readlines()
    base_lines = base_file.readlines()
    assert len(target_lines) == len(base_lines)

    manhattan_distance(target_lines, base_lines)
    euclidean_distance(target_lines, base_lines)
    cosine_distance(target_lines, base_lines)

    # print(euclidean_distances([target_lines, base_lines]))  # 欧氏距离
    print("cosine_distance = ", cosine_distances(
        [target_lines, base_lines]))  # 余弦距离
    # print(cosine_similarity([target_lines, base_lines]))  # cos相似度
