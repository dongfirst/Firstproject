import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
n_classes = 4  # 几种分类的
preteain = False  # 是否下载使用训练参数 有网true 没网false
epoches = 10  # 训练的轮次
traindataset = datasets.ImageFolder(root='./dataset/train/', transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

]))

testdataset = datasets.ImageFolder(root='./dataset/test/', transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

]))

classes = testdataset.classes
print(classes)

model = models.resnet18(pretrained=preteain)
if preteain == True:
    for param in model.parameters():
        param.requires_grad = False
model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
model = model.to(device)


def train_model(model, train_loader, loss_fn, optimizer, epoch):
    model.train()
    total_loss = 0.
    total_corrects = 0.
    total = 0.
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        total_corrects += torch.sum(preds.eq(labels))
        total_loss += loss.item() * inputs.size(0)
        total += labels.size(0)
    total_loss = total_loss / total
    acc = 100 * total_corrects / total
    print("轮次:%4d|训练集损失:%.5f|训练集准确率:%6.2f%%" % (epoch + 1, total_loss, acc))
    return total_loss, acc


def test_model(model, test_loader, loss_fn, optimizer, epoch):
    model.train()
    total_loss = 0.
    total_corrects = 0.
    total = 0.
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            total_loss += loss.item() * inputs.size(0)
            total_corrects += torch.sum(preds.eq(labels))

        loss = total_loss / total
        accuracy = 100 * total_corrects / total
        print("轮次:%4d|训练集损失:%.5f|训练集准确率:%6.2f%%" % (epoch + 1, loss, accuracy))
        return loss, accuracy


loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
train_loader = DataLoader(traindataset, batch_size=32, shuffle=True)
test_loader = DataLoader(testdataset, batch_size=32, shuffle=True)
for epoch in range(0, epoches):
    loss1, acc1 = train_model(model, train_loader, loss_fn, optimizer, epoch)
    loss2, acc2 = test_model(model, test_loader, loss_fn, optimizer, epoch)

classes = testdataset.classes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

path = r'D:\repo_local\Firstproject\resnet18_three\dataset\test\monkey\img_test_895.jpg'  # 测试图片路径
model.eval()
img = Image.open(path)
img_p = transform(img).unsqueeze(0).to(device)
output = model(img_p)
pred = output.argmax(dim=1).item()
p = 100 * nn.Softmax(dim=1)(output).detach().cpu().numpy()[0]
print('该图像预测类别为:', classes[pred])

# 三分类
print('类别{}的概率为{:.2f}%，类别{}的概率为{:.2f}%，类别{}的概率为{:.2f}%，类别{}的概率为{:.2f}%'.format(classes[0], p[0], classes[1], p[1],
                                                                                 classes[2], p[2], classes[3], p[3]))