# module load anaconda3
# activate e1
# pip install numpy
# pip install "monai-weekly[nibabel, tqdm]"
# pip install matplotlib
# %matplotlib inline
# pip install nibabel
# pip install SimpleITK
# pip install helpers
# pip install antspyx

# !pip install antspynet

# !pip install SimpleITK
# 
import os
import glob
import torch
import time
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,random_split
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, EnsureType, Orientation, Spacing,
    RandSpatialCrop, RandFlip, NormalizeIntensity, RandScaleIntensity, RandShiftIntensity
)
from monai.networks.nets import SwinUNETR
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.config import print_config
from monai.transforms import (
    Activations,
    AsDiscrete,
    EnsureTyped,
    MapTransform,
    AsDiscreted,
    Orientationd,
    Spacingd,
    RandSpatialCropd,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ConvertToMultiChannelBasedOnBratsClassesd,
    Orientationd,
    Spacingd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Activations,
    AsDiscrete
)
from torch.utils.data import Subset
import nibabel
import torch.nn.functional as F

from monai.data import Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped

class CustomBraTSDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        image_dir = os.path.join(data_dir, "imagesTr")
        label_dir = os.path.join(data_dir, "labelsTr")

        # Collect all patient IDs
        patient_ids = set()
        for filename in os.listdir(image_dir):
            if filename.endswith(".nii.gz"):
                patient_id = filename.split('_')[0]
                patient_ids.add(patient_id)

        for patient_id in patient_ids:
            self.data.append({
                "image": [
                    os.path.join(image_dir, f"{patient_id}_0001.nii.gz"),
                    os.path.join(image_dir, f"{patient_id}_0001.nii.gz"),
                    os.path.join(image_dir, f"{patient_id}_0002.nii.gz"),
                    os.path.join(image_dir, f"{patient_id}_0003.nii.gz")
                ],
                "label": os.path.join(label_dir, f"{patient_id}.nii.gz")
            })
        super().__init__(self.data, transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform:
            data = self.transform(data)
        return data

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            #Label 1 NEC
            result.append(d[key] == 1)
            #Label 2 SNFH
            result.append(d[key] == 2)
            #Label 3 ET
            result.append(d[key] == 3)
            #Label 4 RC
            result.append(d[key] == 4)
#             # merge label 1 and label 3 to construct TC
#             result.append(torch.logical_or(d[key] == 1, d[key] == 3))
#             # merge label 2, label 1 and label 3 to construct WT
#             result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] ==1))

            d[key] = torch.stack(result, axis=0).float()
        return d


train_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image", "label"], roi_size=[ 182, 218, 182 ], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)

val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)

root_dir = "nnUNet_raw_2024/Dataset001_GLI"
# root_dir="nnUNet"
full_dataset = CustomBraTSDataset(data_dir=root_dir, transform=None)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply transforms to training and validation datasets
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=5)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=5)

max_epochs = 300
val_interval = 1
VAL_AMP = True

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
device = torch.device("cuda")
model = SwinUNETR(
    img_size=(192, 224, 192), 
    in_channels=4,            
    out_channels=4,           
    feature_size=48,
#    drop_rate=0.0,
#    attn_drop_rate=0.0,
#    dropout_path_rate=0.0,
    use_checkpoint=True 
    ).to(device)
loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


# define inference method
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(192, 224, 192 ),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


# use amp to accelerate training
#scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
#torch.backends.cudnn.benchmark = True

best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], [], []]
epoch_loss_values = []
metric_values = []
# metric_values_tc = []
# metric_values_wt = []
metric_values_et = []
metric_values_rc = []
metric_values_snfh = []
metric_values_nec = []

total_start = time.time()
for epoch in range(max_epochs):
    print(device)
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    print("Model in training")
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        print("batch loaded")
        step_start = time.time()
        step += 1
        inputs=batch_data["image"].to(device)
#        batch_data["image"]=batch_data["image"].detach().cpu()
        labels=batch_data["label"].to(device)
#         inputs, labels = (
#             batch_data["image"].to(device),
#             batch_data["label"].to(device),
#         )
#         print("@,", step)
        print(1)
#        with torch.cuda.amp.autocast():
#             print("outputs", outputs.shape)
#             print("label", inputs.shape)
        pad_sizes = (5, 5, 3, 3, 5, 5)
        print(inputs.shape)
        resized_tensor = F.pad(inputs, pad_sizes, mode="constant", value=0).to(device)
        print(2)
        print(resized_tensor.shape)
        outputs = model(resized_tensor)
        print(3)
        outputs = outputs[:, :, 5:-5, 3:-3, 5:-5]
        print(4)
        loss = loss_function(outputs, labels)
        print(5)
            
        print(6)
#        scaler.scale(loss).backward()
#        scaler.step(optimizer)
#        scaler.update()
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            f"{step}/{len(train_dataset) // train_loader.batch_size}"
            f", train_loss: {loss.item():.4f}"
            f", step time: {(time.time() - step_start):.4f}"
        )
    print("@!")
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_data["image"]=val_data["image"].detach().cpu()
                val_data["label"]=val_data["label"].detach().cpu()
                pad_sizes = (5, 5, 3, 3, 5, 5)
                val_inputs = F.pad(val_inputs, pad_sizes).to(device)
                val_outputs = inference(val_inputs)
                val_outputs = val_outputs[:, :, 5:-5, 3:-3, 5:-5]
#                 val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                val_outputs = [post_trans(i) for i in val_outputs]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()

            metric_nec = metric_batch[0].item()
            metric_values_nec.append(metric_nec)

            metric_snfh = metric_batch[1].item()
            metric_values_snfh.append(metric_snfh)

            metric_et = metric_batch[2].item()
            metric_values_et.append(metric_et)

            metric_rc = metric_batch[3].item()
            metric_values_rc.append(metric_rc)

            dice_metric.reset()
            dice_metric_batch.reset()
            
            current_time = time.localtime()
            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join("monai_models", f"best_monai_model_{epoch}_unetr.pth"),
                )
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\n nec: {metric_nec:.4f} snfh: {metric_snfh:.4f} et: {metric_et:.4f} rc: {metric_rc:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f"\n at epoch: {best_metric_epoch}"
            )
            with open('metrics_log_unetr.txt', 'a+') as f:
                f.write(f" current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\n nec: {metric_nec:.4f} snfh: {metric_snfh:.4f} et: {metric_et:.4f} rc: {metric_rc:.4f}"
                f"\n best mean dice: {best_metric:.4f}"
                f"\n at epoch: {best_metric_epoch}\n"
                f"\nepoch end time: {formatted_time}\n")
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start

# print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")

with open('metrics_log_final_unetr.txt', 'a+') as f:
  f.write(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")

