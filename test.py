# 修改 xml文件的 path 为自己的路径。
# 我的图片名称与标注文件名称相同，仅仅后缀不一样（jpg|xml）

import os
import xml.etree.ElementTree as ET

root = 'D:/yolo/yolov5-6.0/data'
ann_path = os.path.join(root, 'Annotations')  # xml文件所在路径

for file_list in os.listdir(ann_path):  # xml 文件列表
    xml_path = os.path.join(ann_path, file_list)
    fname, ext = os.path.splitext(file_list)  # xml文件名
    save_path = root + '/Annotations/' + fname + '.xml'  # 保存修改后的xml的路径
    save_dir = 'D:/yolo/yolov5-6.0/data/images/' + fname + '.jpg'  # 数据集所在相对路径

    tree = ET.parse(xml_path)
    for path in tree.findall('path'):
        path.text = save_dir  # 修改 xml 的 path
        tree.write(save_path)  # 保存修改后的 xml文件 到新路径

