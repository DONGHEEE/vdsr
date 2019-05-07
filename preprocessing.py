import tensorflow as tf
import cv2
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import h5py
import math

# 이미지 잘라내기 : 패치단위로 계산하기 위함

train_size = 80
new_image_input = []
new_image_label = []
path = '/Users/jhkimMultiGpus/Desktop/Test'


def cropping(imgs, modulue, image):
    if np.size(imgs.shape)==3:
        (sheight, swidth,_) = image.shape
        sheight = sheight - np.mod(sheight,modulue)
        swidth = swidth - np.mod(swidth,modulue)
        imgs = imgs[0:sheight,0:swidth,:]
    else:
        (sheight, swidth) = image.shape
        sheight = sheight - np.mod(sheight,modulue)
        swidth = swidth - np.mod(swidth,modulue)
        imgs = imgs[0:sheight,0:swidth]        
    return imgs

# 이미지 저장
def save_img(img, name, title):
    plt.imshow(img)
    plt.title(title)
    plt.savefig(name)
    # cv2.imwrite(name, img)

# 이미지 블러 효과
def gaussian_blur(img):
    imgYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(imgYCrCb, -1, kernel)

def txt_make(filedir,height,width,name,data):
    file = open(filedir + '/txt' + name + '.txt','w')
    for x in range(width):
        for y in range(height):
            write_rgb = "{0} {1}".format(y,x)
            rgb=" {0} {1} {2}\n".format(data[y][x][0], data[y][x][1], data[y][x][2])
            file.write(write_rgb + rgb)
    file.close()

# 이미지 하나당 전처리하는 과정
def preprocessing(full_name, scale, p):
    name = os.path.split(full_name)
    image = cv2.imread(full_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    subname="result/"+"original("+name[1]+").jpg"
    #save_img(image, subname, "original : "+name[1])
    #txt_make("Test/Sequence/input_txt", 64, 64, name[1], image)

    # 논문에서 언급했듯이 컬러 공간 변환 : RGB -> YCrCb
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    image = cropping(image, scale, image)
    im_label = image
    (height, width,_) = im_label.shape

    # 입력 이미지 : 저화질 이미지를 만들기 위한 과정
    im_input = cv2.resize(im_label, (0, 0), fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_CUBIC)
    im_input = cv2.resize(im_input, (0, 0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    im_input = im_input.astype('float32')/255.0
    im_label = im_label.astype('float32')/255.0
    temp_image = (im_input* 255).astype('uint8')
    temp_image = cv2.cvtColor(temp_image, cv2.COLOR_YCrCb2RGB)

    # subname = "/Users/jhkimMultiGpus/Desktop/Test/input/" + "input(" + name[1] + ").jpg"
    subname = "result/last_test/" + "input(" + name[1] + ").jpg"
    #
    save_img(temp_image, subname, "input : "+name[1])


    # 오차 이미지(Residual image)를 만들기 위한 과정
    residual_input = temp_image
    residual_origin = (im_label * 255).astype('uint8')
    residual_origin = cv2.cvtColor(residual_origin, cv2.COLOR_YCrCb2RGB)
    residual_image = residual_origin - residual_input
    subname="result/last_test/" + "residual("+name[1]+").jpg"
    #subname= "/Users/jhkimMultiGpus/Desktop/Test/input/" + "residual(" + name[1] + ").jpg"

    save_img(residual_image, subname, "residual : "+name[1])

    # residual_origin = im_label
    # residual_input = im_input

    # txt_make('/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial/VDSR/result/residual_txt',height,width
    #          ,name[1] ,residual_image)

    # residual_image = (residual_image * 255).astype('uint8')
    # residual_image = cv2.cvtColor(residual_image, cv2.COLOR_YCrCb2RGB)
    # txt_make('/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial/VDSR/result/residual_txt',height,width
    #          ,name[1]+'_rgb_v', residual_image)
    # subname = "result/" + "residual(" + name[1] + ").jpg"
    # save_img(residual_image, subname, "residual : "+name[1])


    # "오차 이미지 + 입력 이미지 = 고품질 이미지"를 확인하는 과정
    capture_image = residual_input + residual_image
    subname="result/last_test/"+"sumImage("+name[1]+").jpg"
    save_img(capture_image, subname, "sumImage : "+name[1])
    #
    # im_input = cropping(im_input,train_size,im_input)
    # im_label = cropping(im_label, train_size, im_label)     #80 cropping

    im_label = im_label.reshape(1,height,width,3)
    im_input = im_input.reshape(1,height,width,3)
    new_image_input.append(im_input)
    new_image_label.append(im_label)

    # (h,w,_) = im_label.shape
    #
    # for i in range(0,h,train_size):
    #     for j in range(0,w,train_size):
    #         im_new = im_input[i: i+train_size, j: j+train_size, 0:3]
    #         im_new_l = im_label[i:i+train_size, j:j+train_size, 0:3]
    #         print(im_new.shape)
    #         new_image_input.append(im_new)
    #         new_image_label.append(im_new_l)
    #         # real_num_r(im_new, train_size, train_size, p, "input")
    #         # real_num_r(im_new_l, train_size, train_size, p, "label")




    # real_num_r(im_input, width, height, i, "input")
    # real_num_r(im_label, width, height, i, "label")


def real_num_r(data,width,height,num,stri):
    data=np.array(data,dtype="float32")
    if stri == "input":
        file = open('/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial/VDSR/Test/Sequence/SequenceText/input/' + 'input_' + str(num) + '.txt', 'w')
    if stri == "label":
        file = open('/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial/VDSR/Test/Sequence/SequenceText/label/' + 'label_' + str(num) + '.txt', 'w')
    # if stri == "input":
    #     file = open('/Users/jhkimMultiGpus/Desktop/Test/input/inputText/' + 'input_Test' + str(num) + '.txt', 'w')
    # if stri == "label":
    #     file = open('/Users/jhkimMultiGpus/Desktop/Test/input/labelText/' + 'label_Test' + str(num) + '.txt', 'w')

    # print(data.shape)
    w_h = str(height)+" "+ str(width)+"\n"
    file.write(w_h)
    for x in range(width):
        for y in range(height):
            write_rgb = "{0} {1}".format(y,x)
            bgr = " {0} {1} {2}\n".format(data[y][x][0], data[y][x][1], data[y][x][2])
            file.write(write_rgb+bgr)
    file.close()

# 폴더 내 이미지를 일괄처리 하기 위한 함수
def batch_preprocessing(dir_path, scale):
    file_names = os.listdir(dir_path)
    for p, filename in enumerate(file_names):
        full_filename = os.path.join(dir_path, filename)
        name, ext = os.path.splitext(full_filename)
        print(full_filename)
        if ext == '.png' or ext == '.bmp' or ext == '.jpg':
            preprocessing(full_filename, scale, p)
    print(new_image_label.__len__(),new_image_input.__len__())

    # savepath = "train_py_forT.h5"
    # new_image_l = np.asarray(new_image_label)
    # new_image_i = np.asarray(new_image_input)
    #
    # with h5py.File(savepath, 'w') as hf:
    #     hf.create_dataset('test_input', data=new_image_i)
    #     hf.create_dataset('test_label', data=new_image_l)
    savepath = "result/last_test.h5"
    with h5py.File(savepath,'w') as hf:
        hf.create_dataset('test_input',data = np.asarray(new_image_input))
        hf.create_dataset('test_label',data = np.asarray(new_image_label))

# batch_preprocessing('Test/Sequence', 4)
# batch_preprocessing('/Users/jhkimMultiGpus/Desktop/Test', 8)
batch_preprocessing('Test/h', 3)


