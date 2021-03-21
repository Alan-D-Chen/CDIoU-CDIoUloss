# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: test.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2021,三月 02
# ---


import time
import os
import json

predictions = [1, 2, 3, 4, 56, 7, 8, 9]

######################################################
stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
json_file_path = os.path.join("/home/alanc/Documents/ATSS/prediction", stamp + "prediction1.json")
# json_file_path = '/home/zxq/PycharmProjects/data/ciga_call/result.json'
json_file = open(json_file_path, mode='a')
"""
save_json_content = []
for img_name in img_name_list:
    result_json = {
        "image_name": img_name,
        "category": 1,
        "score": 0.99074}
    save_json_content.append(result_json)
"""
json.dump(predictions, json_file, indent=4)