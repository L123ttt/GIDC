
import torch
import torch.nn as nn
import torch.nn.functional as F
from lt_data import criteon1
from lt_data import criteon2
from lt_data import show_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(7)


def leaky_relu(x, leak=0.2):
    return F.leaky_relu(x, negative_slope=leak)


class GIDCNet(nn.Module):
    def __init__(self):
        super(GIDCNet, self).__init__()
        c_size = 5
        d_size = 5

        # Encoder
        self.conv0 = nn.ConvTranspose2d(1, 16, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn0 = nn.BatchNorm2d(16)

        self.conv1 = nn.ConvTranspose2d(16, 16, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv1_1 = nn.Conv2d(16, 16, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn1_1 = nn.BatchNorm2d(16)

        self.conv_pooling_1 = nn.Conv2d(16, 16, kernel_size=d_size, stride=2, padding=(d_size - 1) // 2)
        self.bn_pooling_1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn2_1 = nn.BatchNorm2d(32)

        self.conv_pooling_2 = nn.Conv2d(32, 32, kernel_size=d_size, stride=2, padding=(d_size - 1) // 2)
        self.bn_pooling_2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv3_1 = nn.Conv2d(64, 64, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn3_1 = nn.BatchNorm2d(64)

        self.conv_pooling_3 = nn.Conv2d(64, 64, kernel_size=d_size, stride=2, padding=(d_size - 1) // 2)
        self.bn_pooling_3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv4_1 = nn.Conv2d(128, 128, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn4_1 = nn.BatchNorm2d(128)

        self.conv_pooling_4 = nn.Conv2d(128, 128, kernel_size=d_size, stride=2, padding=(d_size - 1) // 2)
        self.bn_pooling_4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv5_1 = nn.Conv2d(256, 256, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn5_1 = nn.BatchNorm2d(256)

        # Decoder
        self.conv6 = nn.ConvTranspose2d(256, 128, kernel_size=c_size, stride=2, padding=(c_size - 1) // 2, output_padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv6_1 = nn.Conv2d(256, 128, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn6_1 = nn.BatchNorm2d(128)

        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.conv7 = nn.ConvTranspose2d(128, 64, kernel_size=c_size, stride=2, padding=(c_size - 1) // 2, output_padding=1)
        self.bn7 = nn.BatchNorm2d(64)

        self.conv7_1 = nn.Conv2d(128, 64, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn7_1 = nn.BatchNorm2d(64)

        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.conv8 = nn.ConvTranspose2d(64, 32, kernel_size=c_size, stride=2, padding=(c_size - 1) // 2, output_padding=1)
        self.bn8 = nn.BatchNorm2d(32)

        self.conv8_1 = nn.Conv2d(64, 32, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn8_1 = nn.BatchNorm2d(32)

        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn8_2 = nn.BatchNorm2d(32)

        self.conv9 = nn.ConvTranspose2d(32, 16, kernel_size=c_size, stride=2, padding=(c_size - 1) // 2, output_padding=1)
        self.bn9 = nn.BatchNorm2d(16)

        self.conv9_1 = nn.Conv2d(32, 16, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn9_1 = nn.BatchNorm2d(16)

        self.conv9_2 = nn.Conv2d(16, 16, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn9_2 = nn.BatchNorm2d(16)

        self.conv10 = nn.Conv2d(16, 1, kernel_size=c_size, stride=1, padding=(c_size - 1) // 2)
        self.bn10 = nn.BatchNorm2d(1)

    def forward(self, x_in, data1, num_patterns):
        # Encoder
        x_in = torch.from_numpy(x_in).float().to(device)
        x0 = leaky_relu(self.bn0(self.conv0(x_in)))
        x1 = leaky_relu(self.bn1(self.conv1(x0)))
        x1_1 = leaky_relu(self.bn1_1(self.conv1_1(x1)))
        Convpool_1 = leaky_relu(self.bn_pooling_1(self.conv_pooling_1(x1_1)))

        x2 = leaky_relu(self.bn2(self.conv2(Convpool_1)))
        x2_1 = leaky_relu(self.bn2_1(self.conv2_1(x2)))
        Convpool_2 = leaky_relu(self.bn_pooling_2(self.conv_pooling_2(x2_1)))

        x3 = leaky_relu(self.bn3(self.conv3(Convpool_2)))
        x3_1 = leaky_relu(self.bn3_1(self.conv3_1(x3)))
        Convpool_3 = leaky_relu(self.bn_pooling_3(self.conv_pooling_3(x3_1)))

        x4 = leaky_relu(self.bn4(self.conv4(Convpool_3)))
        x4_1 = leaky_relu(self.bn4_1(self.conv4_1(x4)))
        Convpool_4 = leaky_relu(self.bn_pooling_4(self.conv_pooling_4(x4_1)))

        x5 = leaky_relu(self.bn5(self.conv5(Convpool_4)))
        x5_1 = leaky_relu(self.bn5_1(self.conv5_1(x5)))

        # # Decoder
        x6 = leaky_relu(self.bn6(self.conv6(x5_1)))
        merge1 = torch.cat([x4_1, x6], dim=1)
        x6_1 = leaky_relu(self.bn6_1(self.conv6_1(merge1)))
        x6_2 = leaky_relu(self.bn6_2(self.conv6_2(x6_1)))

        x7 = leaky_relu(self.bn7(self.conv7(x6_2)))
        merge2 = torch.cat([x3_1, x7], dim=1)
        x7_1 = leaky_relu(self.bn7_1(self.conv7_1(merge2)))
        x7_2 = leaky_relu(self.bn7_2(self.conv7_2(x7_1)))

        x8 = leaky_relu(self.bn8(self.conv8(x7_2)))
        merge3 = torch.cat([x2_1, x8], dim=1)
        x8_1 = leaky_relu(self.bn8_1(self.conv8_1(merge3)))
        x8_2 = leaky_relu(self.bn8_2(self.conv8_2(x8_1)))

        x9 = leaky_relu(self.bn9(self.conv9(x8_2)))
        merge4 = torch.cat([x1_1, x9], dim=1)
        x9_1 = leaky_relu(self.bn9_1(self.conv9_1(merge4)))
        x9_2 = leaky_relu(self.bn9_2(self.conv9_2(x9_1)))

        x10 = torch.sigmoid(self.bn10(self.conv10(x9_2)))

        # Measurement process
        x = x10 / torch.max(x10) * 255
        M = torch.zeros(num_patterns).to(device)
        for i in range(num_patterns):
            pattern = data1['patterns'][:, :, i]
            pattern_tensor = torch.tensor(pattern).to(device)
            pattern_tensor = torch.reshape(pattern_tensor,[1, 1, 64, 64])
            B = (pattern_tensor * x).sum()
            M[i] = B
        y = torch.reshape(M, [1, num_patterns, 1, 1])

        mean_x = torch.mean(x, dim=(0, 1, 2, 3))
        mean_y = torch.mean(y, dim=(0, 1, 2, 3))
        variance_x = torch.var(x, dim=(0, 1, 2, 3), unbiased=False)
        variance_y = torch.var(y, dim=(0, 1, 2, 3), unbiased=False)
        out_x_norm = (x - mean_x.view(1, -1, 1, 1, 1)) / torch.sqrt(variance_x.view(1, -1, 1, 1, 1))
        out_y_norm = (y - mean_y.view(1, -1, 1, 1, 1)) / torch.sqrt(variance_y.view(1, -1, 1, 1, 1))

        return out_x_norm, out_y_norm


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # 假设使用正态分布进行初始化，需要调整为与TensorFlow相似的参数
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 调整std以匹配TensorFlow的truncated_normal
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


net = GIDCNet()
net.apply(weight_init)
net.to(device)

"""
优化模型
"""


def net_train(Steps, inp, img_W, img_H, num_patterns, y_real, optimizer, scheduler, step_count, obj, DGI_temp0, y_real_temp, picture, data1):
    print("start training ...!")
    net.train()
    for step in range(Steps):
        optimizer.zero_grad()

        y_real = torch.Tensor(y_real)
        out_x, out_y = net(inp, data1, num_patterns)

        out_x = torch.reshape(out_x, [1, 1, img_W, img_H])
        out_y = torch.reshape(out_y, [1, num_patterns, 1, 1])

        # 优化模型
        loss = criteon2(out_y, y_real.to(device)) + criteon1(out_x)
        loss.backward()
        optimizer.step()
        scheduler.step()

        print("step: ", step, "   Loss: ", loss.item())

        if step % step_count == 0:
            # 展示图片效果
            show_data(img_W, img_H, num_patterns, out_x, out_y, obj, DGI_temp0, y_real_temp)
    # torch.save(net, f'model_{picture}.pth')  # 保存整个模型
