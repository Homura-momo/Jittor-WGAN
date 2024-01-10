import jittor as jt
from jittor.dataset.cifar import CIFAR10
from jittor.dataset.mnist import MNIST
from jittor import transform
from model import WGAN

import os
import argparse

# 用于可视化
# Windows下cuda版pytorch会与jittor冲突，可在新的conda环境中使用cpu版pytorch
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

# 开启GPU
jt.flags.use_cuda = 1


# %% 输入参数
parser = argparse.ArgumentParser(description="Generic runner for WGAN models")

# 模型参数
parser.add_argument("--latent_dim", type=int, default=64, help="Dimension of the latent space")
parser.add_argument("--load", action="store_true", help="Load saved checkpoints")
parser.add_argument("--no_save", dest="save", action="store_false", help="Save checkpoints")

# 训练参数
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
parser.add_argument("--epoch_num", type=int, default=100, help="Number of epochs to train")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for weights of discriminator")

# 测试参数
parser.add_argument("--test", action="store_true", help="Sample image and not train")
parser.add_argument("--sample_num", type=int, default=100, help="Number of images to sample")

# 数据集参数
parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist", help="Dataset to use")

# 日志参数
parser.add_argument("--log_interval", type=int, default=100, help="Interval for logging training progress")

# 路径参数
parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
parser.add_argument("--log_dir", type=str, default="log/", help="Directory to save log files")
parser.add_argument("--image_dir", type=str, default="img/", help="Directory to save generated images")

args = parser.parse_args()


# %% 设置路径
args.save_dir = os.path.join(args.save_dir, args.dataset)

args.image_dir = os.path.join(args.image_dir, args.dataset)
sample_path = os.path.join(args.image_dir, "sample.png")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
if not os.path.exists(args.image_dir):
    os.makedirs(args.image_dir)

writer = SummaryWriter(os.path.join(args.log_dir, args.dataset))

# %% 加载数据
if args.dataset == "mnist":
    # MNIST
    image_channels = 1
    image_length = 28
    hidden_channels_list = [64, 256]

    # transform.ToTensor: x = x / 255, num_channels:3, Range: (0~255)->(0~1)
    # transform.Gray: x = L / 255, num_channels:1, Range: (0~255)->(0~1)
    # transform.ImageNormalize: x = (x - mean) / std
    # [mean: 0.5, std: 0.5], Range: (0~1)->(-1~1)
    # [mean: mean_x, std: std_x], data mean: 0, data std: 1
    # ImageNormalize将data均值变为0，有助于提高训练的稳定性
    transform_input = transform.Compose([transform.Gray(), transform.ImageNormalize(mean=[0.5], std=[0.5])])

    train_loader = MNIST(train=True, transform=transform_input).set_attrs(batch_size=args.batch_size, shuffle=True)
    val_loader = MNIST(train=False, transform=transform_input).set_attrs(batch_size=args.batch_size, shuffle=False)

elif args.dataset == "cifar10":
    # CIFAR10
    image_channels = 3
    image_length = 32
    hidden_channels_list = [64, 128, 256, 512]

    # transform.ToTensor: x = x / 255, num_channels:3, Range: (0~255)->(0~1), dims: [batch_size, 32, 32, 3]->[batch_size, 3, 32, 32]
    transform_input = transform.Compose([transform.ToTensor(), transform.ImageNormalize(mean=[0.5], std=[0.5])])

    train_loader = CIFAR10(train=True, transform=transform_input).set_attrs(batch_size=args.batch_size, shuffle=True)
    val_loader = CIFAR10(train=False, transform=transform_input).set_attrs(batch_size=args.batch_size, shuffle=False)

# decoder的输出范围为(-1,1)，需要转换到(0,1),以便save_image
transform_output = transform.Compose([transform.ImageNormalize(mean=[-1], std=[2])])


# %% 加载模型
model = WGAN(image_channels, hidden_channels_list, image_length, writer, args)

if args.load:
    model.load()


# %% 训练
def train():
    model.train(train_loader, transform_output)


# %% 采样
def sample():
    print("Sampling...")
    model.eval()

    with jt.no_grad():
        output = model.sample(args.sample_num)
        imgs = transform_output(output).numpy()
        imgs = make_grid(torch.from_numpy(imgs), nrow=10)

        save_image(imgs, sample_path)


# %% 主函数
if __name__ == "__main__":
    if args.test:
        sample()
    else:
        train()
