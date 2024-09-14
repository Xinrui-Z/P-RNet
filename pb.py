import os
import cv2


def file_name(file_dir):
    listName = []
    # for root, dirs, files in os.walk(file_dir):
    for dir in os.listdir(file_dir):
        # print(dir)  # 当前目录路径
        listName.append(dir);
        # print(dirs)  # 当前路径下所有子目录
        # print(files)  # 当前路径下所有非目录子文件
    return listName


jpg_path = 'C:/Users/Administrator/Desktop/images/'

file_dir = 'C:/Users/Administrator/Desktop/5'  # 待取名称文件夹的绝对路径
listname = file_name(file_dir)  # listname 就是需要的标签样本
pre = 'C:/Users/Administrator/Desktop/5/'
j = 1
for video in listname:
    i = 1
    c = 1
    video_path = pre + video
    cap = cv2.VideoCapture(video_path)
    print(video_path)
    timeF = 25
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if (c % timeF == 0):
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imencode('.jpg', frame)[1].tofile(jpg_path + '_' + str(j) + '_' + 'frame_' + str(i) + '.jpg')
            cv2.imshow('frame', frame)
            i += 1
        c = c + 1
        cv2.waitKey(1)
    j += 1
    cap.release()
