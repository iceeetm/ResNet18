import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, random_split
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt, patches as patches
import os

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
batch_size = 2
train_ratio = 0.85
num_classes = 6
learning_rate = 0.01
step_size = 10
epochs = 10

# Define transformations
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset
dataset = datasets.ImageFolder(r'D:\DataSet\dataset\six_vegetables_images', transform=transformations)

# Separate data by class
class_indices = defaultdict(list)
for idx, (_, class_label) in enumerate(dataset.samples):
    class_indices[class_label].append(idx)

# Initialize lists for train and validation indices
train_indices = []
valid_indices = []

# Split data into train and validation sets for each class
for class_label, indices in class_indices.items():
    num_samples = len(indices)
    num_train_samples = int(num_samples * train_ratio)
    
    # Shuffle indices for each class
    torch.manual_seed(42)  # For reproducibility
    shuffled_indices = torch.randperm(num_samples).tolist()
    
    # Divide shuffled indices into train and validation sets
    train_indices.extend([indices[i] for i in shuffled_indices[:num_train_samples]])
    valid_indices.extend([indices[i] for i in shuffled_indices[num_train_samples:]])

# Create Subset datasets for train and validation
train_data = Subset(dataset, train_indices)
valid_data = Subset(dataset, valid_indices)
# print(f"train data num is {len(train_data)}, valid data num is {len(valid_data)}")
# Create data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

# Calculate and print class counts in validation set
class_counts = defaultdict(int)
for data, target in valid_loader:
    for label in target:
        class_counts[label.item()] += 1

# Output class counts
class_names = dataset.classes
for class_index in sorted(class_counts.keys()):
    class_name = class_names[class_index]
    count = class_counts[class_index]
    # print(f"Class {class_index} : {class_name}, valid num is: {count}")
# exit()

class ResNet18_Model(nn.Module):
    def __init__(self):
        super(ResNet18_Model,self).__init__()
        self.block_1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False), # [batch_size,64,112,112]
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1), # [batch_size,64,56,56]
        )
        self.block_2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=64),
        )
        self.block_3=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=64),
        )
        self.block_4=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=128),
        )
        self.block_4_1=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1,stride=2,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=128),
        )
        self.block_5=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=128),
        )
        self.block_6=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=256),
        )
        self.block_6_1=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=1,stride=2,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=256),
        )
        self.block_7=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=256),
        )
        self.block_8=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=512),
        )
        self.block_8_1=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1,stride=2,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=512),
        )
        self.block_9=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=False), # [batch_size,64,112,112]
            nn.BatchNorm2d(num_features=512),
        )
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=1)
        self.fc=nn.Sequential(
            nn.Linear(512,num_classes),
        )
    def forward(self,x):#[6, 3, 224, 224]
        x=self.block_1(x) #[6, 64, 56, 56]
        x=F.relu(self.block_2(x)+x) #[6, 64, 56, 56]
        x=F.relu(self.block_3(x)+x) #[6, 64, 56, 56]
        c2=x
        x=F.relu(self.block_4(x)+self.block_4_1(x)) #[6, 128, 28, 28]
        x=F.relu(self.block_5(x)+x) #[6, 128, 28, 28]
        c3=x
        x=F.relu(self.block_6(x)+self.block_6_1(x)) #[6, 256, 14, 14]
        x=F.relu(self.block_7(x)+x) #[6, 256, 14, 14]
        c4=x
        x=F.relu(self.block_8(x)+self.block_8_1(x)) # [6, 512, 7, 7]
        x=F.relu(self.block_9(x)+x) # [6, 512, 7, 7]
        c5=x
        # print('c5=',c5.shape,'c4=', c4.shape,'c3=', c3.shape,'c2=', c2.shape)
        # quit()
        # c5= torch.Size([6, 512, 7, 7]) c4= torch.Size([6, 256, 14, 14]) c3= torch.Size([6, 128, 28, 28]) c2= torch.Size([6, 64, 56, 56])
        return c2,c3,c4,c5

class FPN_Model(nn.Module):
    def __init__(self):
        super(FPN_Model, self).__init__()
        
        # 定義每個階層的橫向層 (lateral) input C feature maps from backbone to these lateral blocks
        self.c5_block = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False))
        self.c4_block = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False))
        self.c3_block = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False))
        self.c2_block = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False))
        
        # 定義每個階層的輸出層 (output layers) input M feature maps to these blocks, output P feature maps into head
        self.p5_block = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False))
        self.p4_block = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False))
        self.p3_block = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False))
        self.p2_block = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False))

        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest') # up-sample 的不同寫法
        # self.bn = nn.BatchNorm2d(num_features=) 歸一化的寫法

    def forward(self, c2,c3,c4,c5):  # input 4 C feature maps from backbone into leateral blocks, output M feature maps
        # 要從分辨率高 (寬高大的) 的到分辨率低的
        # c2, c3, c4, c5 跟 c5, c4, c3, c2 的順序好像會影響 shape
        # P(n+1) feature maps = M(n+1) up2 + C(n)
        # print('c5=',c5.shape,'c4=', c4.shape,'c3=', c3.shape,'c2=', c2.shape)
        # quit()
        m5 = self.c5_block(c5)
        m4 = self.c4_block(c4) + (F.interpolate(m5, size=c4.size()[2:], mode='nearest'))
        m3 = self.c3_block(c3) + (F.interpolate(m4, size=c3.size()[2:], mode='nearest'))
        m2 = self.c2_block(c2) + (F.interpolate(m3, size=c2.size()[2:], mode='nearest'))

        # p6 = self.maxpool(self.p5_block(m5))
        p5 = self.p5_block(m5)
        p4 = self.p4_block(m4)
        p3 = self.p3_block(m3)
        p2 = self.p2_block(m2)
        # print('c5=', c5.shape,'; c4=', c4.shape,'; c3=', c3.shape,'; c2=', c2.shape)
        # print('m5=', m5.shape,'; m4=', m4.shape,'; m3=', m3.shape,'; m2=', m2.shape)
        # print('p5=', p5.shape,'; p4=', p4.shape,'; p3=', p3.shape,'; p2=', p2.shape)
        # quit()

        # print('p6=', p6.shape)
        # quit()

        return  p2, p3, p4, p5

# 定義 RetinaNet Head 模型
class RetinaNetHead(nn.Module): # anchor-free + decouplehead
    def __init__(self, num_classes, in_channels, feature_size= 256): # in_channels = 輸入特徵圖的通道數, num_classes =  需要分類的類別數量
        super(RetinaNetHead, self).__init__()
        self.num_classes=num_classes

        # 定義分類頭部：卷積層，用於預測每個像素點的類別 = output_channels，這裡是 num_classes
        self.cls_subnet = nn.Sequential(
            nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # 定義回歸頭部：卷積層，用於預測每個像素點的邊界框 (x, y, w, h) = output_channels，這裡是 4
        self.bbox_head = nn.Sequential(
            nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x): # x.shape = [1, 256, 56, 56]
        class_preds = self.classification_head(x) # class_preds.shape = [1, 6, 56, 56] # 通過分類頭部進行前向傳播，獲得類別預測
        bbox_preds = self.regression_head(x) # bbox_preds.shape = [1, 4, 56, 56] # 通過回歸頭部進行前向傳播，獲得邊界框預測
        bbox_preds = self.sigmoid(bbox_preds) # bbox_preds.shape = [1, 4, 56, 56] # 使用 Sigmoid 激活函數將 bbox_preds 限制在 [0, 1] 範圍內
        return class_preds, bbox_preds # 返回類別預測和邊界框預測


# 創建 backbone, FPN neck, head 模型    
backbone_model = ResNet18_Model().to(device)
fpn_model = FPN_Model().to(device)
# head = Head(in_channels=256, num_classes=num_classes).to(device)

image_path = "C:/Users/User/Desktop/color_vegetables/many_kinds_vegetables01.jpg"
original_image = Image.open(image_path).convert("RGB")  # 讀取圖像並進行預處理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 調整圖像大小
    transforms.ToTensor(),          # 轉換為張量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
input_tensor = transform(original_image).unsqueeze(0).to(device)  # 增加一個 batch 維度


# 轉換回 PIL 圖片以供顯示 原始圖片和預處理後的圖片
# preprocessed_image = transforms.ToPILImage()(input_tensor.squeeze(0).cpu())
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# axes[0].imshow(original_image)
# axes[0].set_title('原始圖片')
# axes[1].imshow(preprocessed_image)
# axes[1].set_title('預處理後的圖片')
# plt.show()
# quit()


# print(transform(image).shape) # torch.Size([3, 224, 224])
# print(input_tensor.shape) # torch.Size([1, 3, 224, 224])
# quit()

# 訓練模型
# for i in range(1, epochs+1):
#     print('Running Epochs:' str(i))



c2, c3, c4, c5 = backbone_model(input_tensor)  # 從 backbone 提取特徵圖
fpn_out = fpn_model(c2, c3, c4, c5)  # 將特徵圖輸入 FPN 模型進行融合

# 印 FPN 的輸出特徵圖的形狀
# for i, p in enumerate(fpn_out, start=2):
    # print(f"p{i} shape: {p.shape}")
# p2 shape: torch.Size([1, 256, 56, 56]) p3 shape: torch.Size([1, 256, 28, 28])
# p4 shape: torch.Size([1, 256, 14, 14]) p5 shape: torch.Size([1, 256, 7, 7])

# 印特徵圖中的部分數值
# 這裡只印每個特徵圖的前兩個 batch 的前兩個 channel 的部分數值
# for i, p in enumerate(fpn_out, start=2): # 2 只是命名 從 p2 開始
    # print(f"p{i} sample values: {p[0, :2, :2, :2]}")  # 印第0個樣本的前2個channel中的前 2 行和前 2 列的數值數值
# quit()

# 將融合後的特徵圖輸入 head 進行預測
class_predictions = []  # 存儲每個特徵圖的類別預測結果
bbox_predictions = []  # 存儲每個特徵圖的邊界框預測結果
for fmap in fpn_out: # fmap 代表當前迭代 fpn_out 中的所有特徵圖 # fpn_out 是一個包含多個特徵圖的元組，例如 (p2, p3, p4, p5)
    class_pred, bbox_pred = RetinaNetHead(fmap) # 將 fmap 送入頭部預測 得到 class_pred, bbox_pred
    #class_pred = [batch_size, num_classes, height, width], bbox_pred = [batch_size, 4, height, width]
    class_predictions.append(class_pred) # 加入 list
    bbox_predictions.append(bbox_pred) # 加入 list

# for i, (class_pred, bbox_pred) in enumerate(zip(class_predictions, bbox_predictions), 2):
#     print(f"P{i} class head output shape: {class_pred.shape}") # p2~p5 shape = [batch_size, num_classes, height, weight]
#     # 如 [1, 6, 56, 56], 在 56x56 的特徵圖上，每個位置有 6 個類別預測。
#     print(f"P{i} bbox head output shape: {bbox_pred.shape}") # p2~p5 shape = [batch_size, num_classes, height, weight]
    # [1, 4, 56, 56] 在 56x56 的特徵圖上，每個位置有 4 個邊界框參數 (x, y, w, h) 預測。 (x 和 y)：邊界框的中心點坐標。(w 和 h)：邊界框的寬度和高度。

# 將 bbox 預測轉換為圖像坐標系
def convert_bbox_format(bbox, feature_map_size, original_image_size): # 將預測的邊界框 bbox, 從 feature map, 轉換到圖像上 original image
    print(f"Shape before squeeze: {bbox.shape}")  # 印 squeeze 之前的形狀
    bbox = bbox.squeeze(0)  # 移除批次維度
    print(f"Shape before permute: {bbox.shape}")  # 印 squeeze 之後的形狀
    
    bbox = bbox.permute(1, 2, 0)  # [4, H, W] -> [H, W, 4]
    print(f"Shape after permute: {bbox.shape}")  # 印 permute 之後的形狀
    
    H, W, _ = bbox.shape  # _ 是通道數，這邊是 4
    scale_x = original_image_size[0] / feature_map_size[0]  # original image 和 feature map 的縮放比例
    scale_y = original_image_size[1] / feature_map_size[1]
    
    print(f"Scale x: {scale_x}, Scale y: {scale_y}")  # 印縮放比例

    bbox[:, :, 0] = bbox[:, :, 0] * scale_x  # x 中心點, original x = feature x * scale_x
    bbox[:, :, 1] = bbox[:, :, 1] * scale_y  # y 中心點
    bbox[:, :, 2] = bbox[:, :, 2] * scale_x  # w bbox 寬
    bbox[:, :, 3] = bbox[:, :, 3] * scale_y  # h bbox 高
    
    return bbox  # (x , y , w, h) on image

# 繪製帶有邊界框和標籤的圖像
def show_image_with_boxes(image, class_predictions, bbox_predictions, labels_map, threshold=0.5):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for class_pred, bbox_pred in zip(class_predictions, bbox_predictions):
        class_pred = class_pred.squeeze(0).detach().cpu()
        bbox_pred = bbox_pred.squeeze(0).detach().cpu()
        
        # 取得高於閾值的預測
        scores, labels = torch.max(class_pred, dim=0)
        mask = scores > threshold
        scores = scores[mask]
        labels = labels[mask]
        bbox_pred = bbox_pred[:, mask]
        
        # 繪製邊界框
        for j in range(len(scores)):
            x, y, w, h = bbox_pred[:, j].tolist()
            label = labels[j].item()
            label_name = labels_map.get(int(label), 'Unknown')
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x + w, y, f'{label_name}: {scores[j].item():.2f}', color='black', fontsize=12)  # 调整文本位置到右上角

    plt.show()

# 將圖像轉換回 PIL 格式以供顯示
# image = transforms.ToPILImage()(input_tensor.squeeze(0).cpu())

# 假設 labels_map 是標籤映射
labels_map = {0: 'Lettuce', 1: 'Potato', 2: 'Carrot', 3: 'Onion', 4: 'Garlic', 5: 'Scallion', 6: 'Cauliflower'}

# 顯示圖像並繪製邊界框和標籤
show_image_with_boxes(original_image, class_predictions, bbox_predictions, labels_map)