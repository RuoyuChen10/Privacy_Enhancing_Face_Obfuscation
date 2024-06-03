## 1. 单独调用说明书

进入`MaskFaceGAN`的目录

```python
from models import FaceNet
import torch

facenet = FaceNet(
    net = "VGGFace2",
    include_top = True
)
```

1. 模型已经固定`model.eval()`，不会参与模型更新，如果需要更新请修改[FaceNet.py](./FaceNet.py)第`47`行代码。
2. 模型的权重文件固定，如果移动文件请对应修改[FaceNet.py](./FaceNet.py)第`40`行和第`43`行代码。
3. 模型的输入只需要注意保证输入尺寸shape为`[batch, 3, width, height]`顺序即可，`width`和`height`在模型内部会自动转换。
4. 模型的`device`和输入的`device`需要自己设定
```python
device = "cuda:0"

image = torch.randn([1,3,112,112])
image = image.to(device)

facenet.to(device)
output = facenet(image)
```
5. 模型的参数`net`选择`VGGFace`或`CelebA`，这两个初始化哪个模型在初始化类时候已经固定，不能修改，如果开始需要`VGGFace`，后面需要`CelebA`，请重新初始化一个新类：
```python
# 选择VGGFace2的训练模型
vggfacenet = FaceNet(
    net = "VGGFace2",
    include_top = True
)

# 选择CelebA的训练模型，新初始化一个类
celebanet = FaceNet(
    net = "CelebA",
    include_top = True
)
```
6. 模型是输出预测的类别，还是输出人脸的特征嵌入，是可以实时更正的，也可以在初始化时候就指定。
```python
# 这里输出的是类别

facenet = FaceNet(
    net = "VGGFace2",
    include_top = True
)

image = torch.randn([1,3,112,112])

output = facenet(image)
print("include_top = True, output shape: {}".format(output.shape))

# 这里更改，输出人脸特征表征
model.include_top = False
output = facenet(image)
print("include_top = False, output shape: {}".format(output.shape))
```

输出：

```shell
include_top = True, output shape: torch.Size([1, 8631])
include_top = False, output shape: torch.Size([1, 512])
```

其中针对分类(`include_top = True`)，模型已经对输出经过`softmax`，不需要自己手动增加；针对特征提取(`include_top = False`)，模型已经对输出经过`L2`归一化，不需要手动继续增加。

## 2. 配合程序调用说明书

参数请修改[config.yml](../../config.yml)中
```yaml
MODELS:
  SEGMENTATOR:
    N_CLASSES: 9
    CHANGE_EYEBROWS: False
    CKPT: 'FaceParser.ckpt'
    UPDATE_SHAPE: False
  CLASSIFIER:
    CKPT: 'BranchedTiny.ckpt'
  FACENET:
    Net: "VGGFace2" # "VGGFace2" or "CelebA"
    TOP: True
```
最后两行，`Net`可选择`VGGFace2`和`CelebA`，如果输入错误会报错。

已在[model_module.py](../../model_module.py)中的`NewModelsModule`类中更改，定义私有变量为`face_net`。

调用案例：
```python
from model_module import NewModelsModule

models = NewModelsModule(cfg.MODELS, attribute_subset=['Male', 'Female', 'Young', 'Middle Aged', 'Senior','Asian', 'White', 'Black','Black Hair']).to(cfg.DEVICE)

image = torch.randn([1,3, 112,112])   # 输入大小不需要修正，程序自动修正，只要保证通道顺序没错就行

output = self.models.face_net(image)
# shape: torch.Size([1,8631])

# 更改输出是特征：
self.models.face_net.include_top = False
output = self.models.face_net(image)
# shape: torch.Size([1,512])
```