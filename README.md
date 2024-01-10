# Jittor—GAN

清华大学“媒体计算”课程大作业，基于jittor实现WGAN

## Reference

- https://github.com/Jittor/JGAN/blob/master/models/wgan/wgan.py
- https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_clipping.py

## Usage

### 训练

```bash
python main.py --dataset mnist --epoch 100 --learning_rate 1e-4 --batch_size 64
```

更多参数请参考`main.py`

### 测试

```bash
python main.py --test
```

### Tensorboard

```bash
tensorboard --logdir=log
```