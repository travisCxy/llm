import json
import os
import random

import cv2
import numpy as np
from glob import glob


def lefttop_rightbottom_theta_to_4points(region, theta):
    x1, y1, x2, y2 = region
    points = []
    points.append((x1, y1, 1))
    points.append((x2, y1, 1))
    points.append((x2, y2, 1))
    points.append((x1, y2, 1))

    M = cv2.getRotationMatrix2D((x1, y1), - theta, 1)
    Mt = np.transpose(M)
    roated_points = np.matmul(points, Mt)
    ret = []
    for i in range(roated_points.shape[0]):
        ret.append(tuple(roated_points[i]))
    return ret


def get_items(regions, min_x, max_x):
    items = []
    for item in regions:
        if min_x <= item["region"][0] <= max_x and min_x <= item["region"][2] <= max_x:
            items.append(item)
        elif item["region"][0] <= min_x and item["region"][2] >= max_x:
            if (min_x - item["region"][0]) + (item["region"][2] - max_x) < (
                    item["region"][2] - item["region"][0]) * 0.2:
                items.append(item)
        elif item["region"][0] <= min_x and min_x < item["region"][2] < max_x:
            if (min_x - item["region"][0]) < (item["region"][2] - item["region"][0]) * 0.2:
                items.append(item)
        elif min_x < item["region"][0] < max_x and item["region"][2] >= max_x:
            if (item["region"][2] - max_x) < (item["region"][2] - item["region"][0]) * 0.2:
                items.append(item)
    return items


def split(ret):
    topics = [item for item in ret["regions"] if 4 <= item["cls"] <= 9]
    if len(topics) == 0:
        return []
    topics.sort(key=lambda x: x["region"][1])
    ts_list = [[] for i in range(len(topics))]
    for i in range(len(topics)-1):
        for j in range(i+1, len(topics)):
            if topics[j]["region"][0] > topics[i]["region"][2] or topics[j]["region"][2] < topics[i]["region"][0]:
                ts_list[i].append(j)
                ts_list[j].append(i)
    if all([len(ts) == 0 for ts in ts_list]):
        min_x = min([topic["region"][0] for topic in topics])
        max_x = max([topic["region"][2] for topic in topics])
        items = get_items(ret["regions"], min_x, max_x)
        return [{"regions": items}]
    used = [False for i in range(len(topics))]
    columns = []
    for i in range(len(topics)):
        if used[i]:
            continue
        column = [i]
        used[i] = True
        for j in range(i+1, len(topics)):
            if used[j]:
                continue
            if ts_list[j] == ts_list[i]:
                column.append(j)
                used[j] = True
        columns.append(column)
    rets = []
    for column in columns:
        min_x = min([topics[k]["region"][0] for k in column])
        max_x = max([topics[k]["region"][2] for k in column])
        items = get_items(ret["regions"], min_x, max_x)
        rets.append({"regions": items})
    return rets


if __name__ == "__main__":
    from PIL import Image
    # with open('/home/ateam/wu.changqing/qms_files.txt', 'r', encoding='utf-8') as f:
    #     files = f.read().split('\n')
    with open('/mnt/ceph/15/datasets/yyt_det/20210602/YYT_DET_20210602/train.txt', 'r', encoding='utf-8') as f:
        files = f.read().split('\n')
    # random.shuffle(files)
    for path in files:
        # path = os.path.join('/home/ateam/wu.changqing/qms', path)
        # json_path = path[:-4] + '.json'
        path = os.path.join('/mnt/ceph/15/datasets/yyt_det/20210602/YYT_DET_20210602/', path)
        json_path = path[:-4] + '.txt'
        if not os.path.exists(json_path):
            continue
        print(json_path)
        data = json.load(open(json_path, encoding='utf-8'))
        img = cv2.imread(path)
        # if img.shape[0] > img.shape[1]:
        #     continue
    # while True:
    #     img = cv2.imread('2178c4362f830bdbe13cfb075c5b9d6f.jpg')
    #     data = json.load(open('cls.json', encoding='utf-8'))
    #     rets = split(data)
        skip = True
        for item in data['regions']:
            if item['cls'] == 10:
                skip = False
                break
        if skip:
            continue
        for item in data['regions']:
            # if item['cls'] != 3:
            #     continue
            # print(item['cls'], item['result'], item['rotation'])
            # print(item['rotation'])
            region = item['region']
            threa = item['rotation']
            points = lefttop_rightbottom_theta_to_4points(region, threa)
            x0 = int(min(point[0] for point in points))
            y0 = int(min(point[1] for point in points))
            x1 = int(max(point[0] for point in points))
            y1 = int(max(point[1] for point in points))
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.putText(img, str(item['cls']), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        img = cv2.resize(img, (1000, 1000))
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break
