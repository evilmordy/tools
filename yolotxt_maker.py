import cv2 as cv
import os
import numpy as np
import pandas as pd
from tkinter import Tk,filedialog,simpledialog



def get_input():
    global image
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    image = filedialog.askopenfilename(
        title="选择图像文件",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
    )
    root.destroy()

    if not image:  # 如果用户取消了选择
        exit()

    global class_id

    root2  = Tk()
    root2.withdraw()

    class_id = simpledialog.askstring("输入","请输入class id>>")

    if not class_id:
        exit()

    root2.destroy()

get_input()
img = cv.imread(image,1)
img_copy = img.copy()
clone = None
drawing = False
begin_x=-1
begin_y=-1
end_x=-1
end_y=-1
name = 'YOLO txt maker'
img_name = os.path.splitext(os.path.basename(image))[0]
txt_name  =f'{img_name}.txt'

def draw_rect(event,x,y,flags,param):
    global begin_x,begin_y,end_x,end_y,drawing,img,clone,num

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        begin_x = x
        begin_y = y
        end_x = x
        end_y = y
        clone = img.copy()

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            img = clone.copy()
            end_x = x
            end_y = y
            cv.rectangle(img,(begin_x,begin_y),(end_x,end_y),(100,200,0),2)
            cv.imshow(name,img)


    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        end_x = x
        end_y = y
        cv.rectangle(img,(begin_x,begin_y),(end_x,end_y),(0,0,255),2)
        cv.imshow(name,img)
        center_x = (begin_x+end_x) / (2 * img.shape[1])
        center_y = (begin_y+end_y)/ (2 * img.shape[0])
        width = abs(end_x-begin_x)/img.shape[1]
        height = abs(end_y-begin_y)/img.shape[0]
        data = [class_id ,center_x, center_y, width, height]
        with open(txt_name, 'w', encoding='utf-8') as f:
            for value in data:
                f.write(f'{value} ')




cv.namedWindow(name)
cv.setMouseCallback(name,draw_rect)
cv.imshow(name,img)

while True:
    key = cv.waitKey(1) & 0xFF

    if key == ord('r'):
        img = img_copy.copy()

        cv.imshow(name, img)

    elif key == ord('k'):
        cv.destroyAllWindows()
        get_input()
        img = cv.imread(image, 1)
        img_copy = img.copy()
        begin_x = -1
        begin_y = -1
        end_x = -1
        end_y = -1
        img_name = os.path.splitext(os.path.basename(image))[0]
        txt_name = f'{img_name}.txt'
        cv.namedWindow(name)
        cv.setMouseCallback(name, draw_rect)
        cv.imshow(name, img)
    elif key==ord('q'):
        break

cv.destroyAllWindows()


