# -*-coding:utf-8-*-
import os
import json
import shutil
import pdb

fin = open("conversations_tiku_box.json")
data = json.load(fin)
cnt = 0
for file in data:
    image = file["image"]
    shutil.copyfile(image, os.path.basename(image))
    cnt  += 1
    if cnt > 10:
        break