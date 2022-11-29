import os
import pickle

import dicom2nifti
import nibabel as nib  # 为了处理NIFTI格式的医学方面的数据
import numpy as np
import SimpleITK as sitk

modalities = ('flair', 't1ce', 't1', 't2')

# train
train_set = {
        #'root': 'path to training set',
        'root': '/data/ljf/train/AP0/',
        #'flist': 'all.txt',
        'flist': 'train.txt',#########################################
        'has_label': True
        }

# test/validation data
valid_set = {
        'root': 'path to validation set',
        'flist': 'valid.txt',
        'has_label': False
        }

test_set = {
        'root': 'path to testing set',
        'flist': 'test.txt',
        'has_label': False
        }


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    #import nibabel as nib是为了读取nii格式的文件
    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data
#注释和解除注释ctrl+/
# def readfile():
#     original_path_dicom='/data/ljf/AP/'
#     dicom_file_name=os.listdir(original_path_dicom)
#     dicom_file_name.sort()
#     return dicom_file_name
#--------------------------------------------------------------------------------结果很离奇 
def dcm2nii(dcms_path, nii_path):
	# 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    #print(dicom_names)  #怎么去改变这个文件序列
    image2 = reader.Execute()
	# 2.将整合后的数据转为array，并获取dicom文件基本信息
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z
	# 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, nii_path)



def process_i16(path, has_label=True):
    """ Save the original 3D MRI images with dtype=int16.
        Noted that no normalization is used! """
    label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')

    images = np.stack([
        np.array(nib_load(path + modal + '.nii.gz'), dtype='int16', order='C')
        for modal in modalities], -1)# [240,240,155]

    output = path + 'data_i16.pkl'

    with open(output, 'wb') as f:
        print(output)
        print(images.shape, type(images), label.shape, type(label))  # (240,240,155,4) , (240,240,155)
        pickle.dump((images, label), f)

    if not has_label:
        return


# def process_f32b0(path, has_label=True):
#     """ Save the data with dtype=float32.
#         z-score is used but keep the background with zero! """
#     if has_label:
#         label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')#get absolute path
#     images = np.stack([np.array(nib_load(path + modal + '.nii.gz'), dtype='float32', order='C') for modal in modalities], -1)  # [240,240,155]
#     #images are the paths pf four dimensions files
#     output = path + 'data_f32b0.pkl'
#     mask = images.sum(-1) > 0
#     for k in range(4):

#         x = images[..., k]  #
#         y = x[mask]

#         # 0.8885
#         x[mask] -= y.mean()
#         x[mask] /= y.std()

#         images[..., k] = x

#     with open(output, 'wb') as f:
#         print(output)

#         if has_label:
#             pickle.dump((images, label), f)
#         else:
#             pickle.dump(images, f)

#     if not has_label:
#         return

def process_f32b0(path, has_label=True):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    if has_label:
        label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')#get absolute path
    images = np.stack([np.array(nib_load(path + modal + '.nii.gz'), dtype='float32', order='C') for modal in modalities], -1)  # [240,240,155]
    #images are the paths of four dimensions files
    output = path + 'data_f32b0.pkl'
    mask = images.sum(-1) > 0#一堆bool
    for k in range(4):

        x = images[..., k]  #
        y = x[mask]

        # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[..., k] = x

    with open(output, 'wb') as f:
        print(output)

        if has_label:
            pickle.dump((images, label), f)
        else:
            pickle.dump(images, f)

    if not has_label:
        return

def doit(dset):
    root, has_label = dset['root'], dset['has_label']#has_label==true
    file_list = os.path.join(root, dset['flist'])
    #这个函数用来拼接文件路径，可以#file_list=/data/ljf/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/train.txt
    subjects = open(file_list).read().splitlines()#返回一个列表，没有\n换行符，只是文件路径

    names = [sub.split('/')[-1] for sub in subjects]#names=文件名字，BraTS19_2013_10_1就像train.txt里没有HGG和LGG
    paths = [os.path.join(root, sub, name + '_') for sub, name in zip(subjects, names)]
    #现在整个这个路径就是在凑具体每一个nii文件的名字，process_f32b0里面有体现，has_label得到seg.nii.gz文件绝对路径
    #root='/data/ljf/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/';
    #sub=
    for path in paths:

        process_f32b0(path, has_label)



if __name__ == '__main__':
    # doit(train_set)
    dcms_path = r'/data/ljf/train/AP0/'  # dicom序列文件所在路径
    # name=dcm2nii()
    nii_path = r'/data/ljf/train/AP0/seg.nii.gz'  # 所需.nii.gz文件保存路径

    #readdcm(dcms_path,nii_path)
    #dcm2nii(dcms_path, nii_path)  #最后结果是圆圈的代码
    #不用doit来做，直接process_f32b0
    process_f32b0(dcms_path,True)

    # path='/data/ljf/train/AP0/data_f32b0.pkl'
    # f = open(path,'rb')
    # data = pickle.load(f)

    # print(data)
    # print(len(data))

#下面是为了说明与pkl读数据方式无关的代码
    # wirtedata = {'a':[1.222,2.3333,3.4444],'b':('string','abc'),'c':'hello'}
    # pic = open('/data/ljf/train/AP0/test.pkl','wb')
    # pickle.dump(wirtedata,pic)
    # pic.close()

    # f=open('/data/ljf/train/AP0/test.pkl','rb')
    # data = pickle.load(f)
    # print(wirtedata)
    # print('\n')
    # print(data)

    #doit(valid_set)
    # doit(test_set)

