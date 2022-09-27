import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=3), nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(data_iter)).device

    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, lr, num_epochs, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on ', device)
    net.to(device)
    optimizer = torch.optim.SGD(params=net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_batches):
        metric = d2l.Accumulator(3)  # 训练损失之和，训练准确率之和，样本总数
        net.train()
        for i, (x, y) in enumerate(train_iter):
            timer.start()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(x)
            loss_ = loss(y_hat, y)
            loss_.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss_ * x.shape[0], d2l.accuracy(y_hat, y), x.shape[0])
            timer.stop()
            train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add((epoch + (i + 1) / num_batches), (train_loss, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train_loss{train_loss:.3f},train_acc{train_acc:.3f},'
          f'test_acc{test_acc:.3f}')
    print(f'{metric[2] * num_batches / timer.sum():.1f} examples/sec'
          f'on {str(device)}')


lr = 0.9
num_epochs = 10
train_ch6(net, train_iter, test_iter, lr, num_epochs, d2l.try_gpu())
plt.show()
