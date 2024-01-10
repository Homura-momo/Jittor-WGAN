"""
Jittor——GAN
ref: https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_clipping.py
ref: https://github.com/Jittor/JGAN/blob/master/models/wgan/wgan.py
"""

import jittor as jt
from jittor import nn, optim
from tqdm import tqdm

# 用于可视化
# Windows下cuda版pytorch会与jittor冲突，可在新的conda环境中使用cpu版pytorch
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

import os


class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels_list, image_length):
        """
        :param in_channels: 输入图像的通道数
        :param hidden_channels_list: 中间层的通道数
        """

        super().__init__()

        # 卷积层
        self.conv = nn.Sequential()

        cur_channels = in_channels

        for hidden_channels in hidden_channels_list:
            # 卷积，将图像缩小
            # 卷积后的图像大小计算公式为：N=(W−F+2P)//S​+1
            # 其中 W 表示输入图像的大小，F 表示卷积核的大小，S 表示步长，P 表示填充的像素数
            self.conv.append(nn.Conv2d(cur_channels, hidden_channels, kernel_size=3, stride=2, padding=1))
            self.conv.append(nn.BatchNorm2d(hidden_channels))
            self.conv.append(nn.LeakyReLU())

            # in_channels更新为上一层的输出通道数
            cur_channels = hidden_channels

        # 若F=3，S=2，P=1，则每次卷积后的图像大小缩小一半
        image_length //= 2 ** len(hidden_channels_list)

        # 全连接层，映射为一个值
        self.fc = nn.Linear(cur_channels * image_length**2, 1)

    def execute(self, input):
        """
        判别过程，图像->判别结果
        :param input: (Var) [B x C x H x W] Range: [-1, 1]
        :return: (Var) [B x 1]
        """
        result = self.conv(input)

        # 将多个通道的特征图展平为一维向量
        # [B x C x H x W] -> [B x C*H*W]
        result = jt.flatten(result, start_dim=1)

        result = self.fc(result)

        return result


class Generator(nn.Module):
    def __init__(self, out_channels, latent_dim, hidden_channels_list, image_length):
        """
        :param out_channels: 输出图像的通道数
        :param hidden_channels_list: 中间层的通道数
        :param latent_dim: 噪声Z的维度
        """

        super().__init__()

        # 卷积层
        self.convT = nn.Sequential()

        # 转置卷积会将图像放大，所以输入转置卷积的图像应比输出图像小
        image_length = image_length // (2 ** (len(hidden_channels_list)))

        # 输入转置卷积的图像的[通道数,图像高，图像宽]
        self.convT_input_chw = (hidden_channels_list[0], image_length, image_length)

        # 将噪声从latent_dim映射到更大维度，之后再reshape为多个通道的特征图
        self.projection = nn.Linear(latent_dim, hidden_channels_list[0] * image_length**2)

        for i in range(len(hidden_channels_list) - 1):
            # 转置卷积，将图像放大
            # 转置卷积后的图像大小计算公式为：N=(W−1)∗S−2P+F+Pout
            # 其中 W 表示输入图像的大小，F 表示卷积核的大小，S 表示步长，P 表示填充的像素数，Pout 表示输出填充的像素数
            self.convT.append(nn.ConvTranspose2d(hidden_channels_list[i], hidden_channels_list[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1))
            self.convT.append(nn.BatchNorm2d(hidden_channels_list[i + 1]))
            self.convT.append(nn.LeakyReLU())

        # 将图像通道数变为out_channels
        self.convT.append(nn.ConvTranspose2d(hidden_channels_list[-1], out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))

        # 输出前使用tanh激活，将输出值限制在[-1,1]之间
        self.output = nn.Tanh()

    def execute(self, z):
        """
        生成过程，噪声->图像
        :param z: (Var) [B x D]
        :return: (Var) [B x C x H x W] Range: [-1, 1]
        """
        # 噪声扩充维度
        result = self.projection(z)

        # 从一维向量变形为多个通道的特征图
        result = result.view(-1, *self.convT_input_chw)

        # 放大为图像
        result = self.convT(result)
        result = self.output(result)

        return result


class WGAN(nn.Module):
    def __init__(self, image_channels, hidden_channels_list, image_length, writer, args):
        """
        :param hidden_channels_list: 各个卷积层的通道数，从小到大
        :param args: 参数
        """
        super().__init__()

        self.latent_dim = args.latent_dim
        self.clip_value = args.clip_value
        self.epoch_num = args.epoch_num
        self.n_critic = args.n_critic

        self.writer = writer
        self.need_save = args.save
        self.save_dir = args.save_dir
        self.image_dir = args.image_dir

        self.generator = Generator(image_channels, self.latent_dim, hidden_channels_list[::-1], image_length)
        self.discriminator = Discriminator(image_channels, hidden_channels_list, image_length)

        # 优化器
        # 根据原论文经验，不能使用基于动量的优化器
        self.optimizer_G = optim.RMSprop(self.generator.parameters(), lr=args.learning_rate)
        self.optimizer_D = optim.RMSprop(self.discriminator.parameters(), lr=args.learning_rate)

    def execute(self):
        pass

    def generate(self, num_samples):
        """
        生成图像
        :param num_samples: 生成图像的数量
        :param args: 训练参数
        """
        # 从N(0, 1)中采样得到隐向量，然后解码为图像
        z = jt.randn(num_samples, self.latent_dim)

        result = self.generator(z)
        return result

    def train(self, train_loader, transform_output):
        """
        训练
        :param train_loader: 训练数据集
        :param transform_output: 变换输出图像
        """

        # 固定噪声，用于验证生成器的效果
        fixed_noise = jt.randn(100, self.latent_dim)

        for epoch in range(self.epoch_num):
            super().train()

            for batch_idx, (real_imgs, _) in tqdm(enumerate(train_loader)):
                images_num = real_imgs.shape[0]

                # 训练生成器

                # 生成假图像
                z = jt.randn(images_num, self.latent_dim)

                # 梯度不需要传回生成器
                fake_imgs = self.generator(z).detach()

                D_real = self.discriminator(real_imgs)
                D_G_z = self.discriminator(fake_imgs)

                D_loss = jt.mean(D_G_z) - jt.mean(D_real)

                self.optimizer_D.step(D_loss)

                # 权重裁剪，保证lipshitz连续
                for param in self.discriminator.parameters():
                    # 原代码：param=jt.clamp(param, -self.clip_value, self.clip_value)
                    # 应使用assign，使用=赋值只会更改param所指向的对象，不会改变param所指对象的值
                    param.assign(jt.clamp(param, -self.clip_value, self.clip_value))

                # 每训练n_critic次判别器，训练一次生成器
                # 保证判别器的效果
                if (batch_idx % self.n_critic) == 0:
                    # 训练生成器
                    z = jt.randn(images_num, self.latent_dim)
                    fake_imgs = self.generator(z)

                    D_G_z = self.discriminator(fake_imgs)

                    G_loss = -jt.mean(D_G_z)

                    self.optimizer_G.step(G_loss)

                    batch_num = epoch * len(train_loader) + batch_idx

                    # 记录训练过程
                    print(f"epoch: {epoch}, batch: {batch_idx},D loss: {D_loss.item():.6f}, G loss: {G_loss.item():.6f}")

                    self.writer.add_scalar("D_loss", D_loss.item(), batch_num)
                    self.writer.add_scalar("G_loss", G_loss.item(), batch_num)

            # 每个epoch测试一次生成器效果
            super().eval()
            with jt.no_grad():
                # 将固定噪声解码为图像
                output = self.generator(fixed_noise)

                imgs = transform_output(output).numpy()

                # make_grid: 将多张图片拼接成一张图片网格
                imgs = make_grid(torch.from_numpy(imgs), nrow=10)

                self.writer.add_image("Image", imgs, epoch)

                # 保存图片
                # imgs需满足 shape:[C, H, W], Range: (0~1)
                # C=1为灰度图，C=3为彩色图
                save_image(imgs, os.path.join(self.image_dir, f"epoch_{epoch}.png"))

        if self.need_save:
            self.save()
            print("model saved")

    # 保存模型
    def save(self):
        self.discriminator.save(os.path.join(self.save_dir, "discriminator.pkl"))
        self.generator.save(os.path.join(self.save_dir, "generator.pkl"))
        jt.save(os.path.join(self.save_dir, "optimizer_D.pkl"))
        jt.save(os.path.join(self.save_dir, "optimizer_G.pkl"))

    # 加载模型
    def load(self):
        self.discriminator.load(os.path.join(self.save_dir, "discriminator.pkl"))
        self.generator.load(os.path.join(self.save_dir, "generator.pkl"))
        self.optimizer_D.load_state_dict(os.path.join(self.save_dir, "optimizer_D.pkl"))
        self.optimizer_G.load_state_dict(os.path.join(self.save_dir, "optimizer_G.pkl"))
