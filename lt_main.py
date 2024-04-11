import os
import torch
from torch.optim import optimizer

from lt_unet import net
from lt_unet import net_train
from lt_data import data
# from test import eval_adversarial_samples


def main():
    print("start ...!")
    # 准备数据
    A_real, DGI_temp0, y_real, y_real_temp, inp, obj, data1 = data(img_W, img_H, batch_size, num_patterns, picture_path, path1)
    # # 模型训练
    net_train(Steps, inp, img_W, img_H, num_patterns, y_real, optimizer, scheduler, step_count, obj, DGI_temp0, y_real_temp, picture, data1)
    print('Finished!')


if __name__ == "__main__":
    """
        定义程序的详细参数
    """
    img_W = 64                                          # 宽度
    img_H = 64                                          # 高度
    batch_size = 1                                      # 批量大小
    lr0 = 0.05                                         # 学习率
    Steps = 1001                                        # 训练步骤
    step_count = 100                                    # 结果展示的间隔步骤

    # SR = 0.25                                         # 采样率
    # num_patterns = int(np.round(img_W*img_H*SR))      # 测量次数
    num_patterns = 1000                                 # 测量次数

    attack_loss = "l_2"

    path1 = '参考光'
    path = r"train_CelebA_128/"
    picture = "a.jpg"
    picture_path = path + picture
    result_save_path = '.\\tp\\'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr0, betas=(0.5, 0.9), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    main()
