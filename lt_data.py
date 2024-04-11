import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


"""
定义计算方式
"""


# 计算 psnr 的函数
def eval_psnr(img1_array, img2_array):
    psnr_value = compare_psnr(img1_array, img2_array)
    return psnr_value


# 计算SSIM的函数
def eval_ssim(img1_array, img2_array):
    img1 = np.reshape(img1_array.detach().numpy(), [64,64], order='F')
    img2 = np.reshape(img2_array.detach().numpy(), [[64,64]], order='F')
    ssim_value = ssim(img1, img2)  # 对于灰度图像，multichannel应设为False
    print(ssim_value)


# 定义损失函数
class Loss(torch.nn.Module):
    def forward(self, x, y):
        loss = torch.mean((y - x) ** 2)
        return loss


class TVLoss(torch.nn.Module):
    def __init__(self, TVLoss_weight=1e-9):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


criteon1 = TVLoss()
criteon2 = Loss()


"""
定义数据预处理
"""


def data(img_W, img_H, batch_size, num_patterns, picture_path, path1):
    print('start preparation data ...!')

    data1 = np.zeros([img_W, img_H, 3000])
    for root, dirs, files in os.walk(path1):
        for file in files:
            # for i in range(num_patterns):
            if file.endswith('.png') or file.endswith('.PNG'):
                img = Image.open(os.path.join(root, file))
                img = img.convert('L')
                img = img.resize((img_W, img_H), Image.LANCZOS)
                img = np.array(img)
                data1[:, :, files.index(file)] = img
            # break
    data1 = {'patterns': data1}
    A = Image.open(picture_path)
    A = A.resize((img_W, img_H), Image.LANCZOS)
    A = A.convert("L")
    obj = np.array(A)
    # obj = obj.astype(np.float32) / 255.0

    print('DGI reconstruction ...！')
    M = []
    B_aver = 0
    SI_aver = 0
    R_aver = 0
    RI_aver = 0
    count = 0
    for i in range(num_patterns):
        pattern = data1['patterns'][:, :, i]
        B = sum(sum(np.multiply(obj, pattern)))
        M.append(B)
        B_r = B
        count = count + 1
        SI_aver = (SI_aver * (count - 1) + pattern * B_r) / count
        B_aver = (B_aver * (count - 1) + B_r) / count
        R_aver = (R_aver * (count - 1) + sum(sum(pattern))) / count
        RI_aver = (RI_aver * (count - 1) + sum(sum(pattern)) * pattern) / count
    DGI = SI_aver - B_aver / R_aver * RI_aver

    M = np.array(M)
    M = M.reshape([num_patterns, 1])
    A_real = data1['patterns'][:, :, 0:num_patterns]  # illumination patterns
    y_real = M[0:num_patterns]
    if (num_patterns > np.shape(data1['patterns'])[-1]):
        raise Exception('Please set a smaller SR')  # 如果测量次数过大，输出请减小SR

    # for i in range(num_patterns):
    #     pattern = data1['patterns'][:, :, i]
    #     B_r = M[i]
    #     count = count + 1
    #     SI_aver = (SI_aver * (count - 1) + pattern * B_r) / count
    #     B_aver = (B_aver * (count - 1) + B_r) / count
    #     R_aver = (R_aver * (count - 1) + sum(sum(pattern))) / count
    #     RI_aver = (RI_aver * (count - 1) + sum(sum(pattern)) * pattern) / count
    # DGI = SI_aver - B_aver / R_aver * RI_aver

    y_real = np.reshape(y_real, [batch_size, num_patterns, 1, 1])
    A_real = np.reshape(A_real, [num_patterns, batch_size, img_W, img_H])
    DGI = np.reshape(DGI, [batch_size, 1, img_W, img_H])
    DGI = (DGI - np.mean(DGI)) / np.std(DGI)
    DGI_temp0 = np.reshape(DGI, [img_W, img_H], order='F')
    y_real = (y_real - np.mean(y_real)) / np.std(y_real)
    A_real = (A_real - np.mean(A_real)) / np.std(A_real)
    A_real = torch.tensor(A_real).float()
    y_real_temp = np.reshape(y_real, [num_patterns])
    # y_real = torch.Tensor(y_real)
    # inp = torch.from_numpy(DGI).float()
    inp = DGI

    return A_real, DGI_temp0, y_real, y_real_temp, inp, obj, data1


"""
定义绘图内容
"""

def show_data(img_W, img_H, num_patterns, out_x, out_y, obj, DGI_temp0, y_real_temp):

    x_out = np.reshape(out_x.detach().cpu().numpy(), [img_W, img_H], order='F')
    y_out = np.reshape(out_y.detach().cpu().numpy(), [num_patterns], order='F')

    plt.subplot(151)
    plt.imshow(obj, cmap='gray')
    plt.title('original')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(152)
    plt.imshow(DGI_temp0, cmap='gray')
    plt.title('DGI')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(153)
    plt.imshow(x_out, cmap='gray')
    plt.title('GIDC')
    plt.xticks([])
    plt.yticks([])

    ax1 = plt.subplot(154)
    plt.plot(y_out)
    plt.title('pred_y')
    ax1.set_aspect(1.0 / ax1.get_data_ratio(), adjustable='box')
    plt.yticks([])

    ax2 = plt.subplot(155)
    plt.plot(y_real_temp)
    plt.title('real_y')
    ax2.set_aspect(1.0 / ax2.get_data_ratio(), adjustable='box')
    plt.yticks([])

    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    plt.show()
    plt.close()
