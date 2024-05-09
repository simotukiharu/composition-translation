import sys, os, time
sys.dont_write_bytecode = True
import torch
import torchvision
from torch.utils.data import DataLoader
from pyt_det.engine import train_one_epoch, evaluate
import config as cf
import load_dataset_annot as ld

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
id_str = sys.argv[1]
img_dir_path = sys.argv[2]
annot_file_name = sys.argv[3]
path_log = "_l_" + id_str + ".csv" # loss推移の記録ファイル
output_dir = "_log_" + id_str # 保存用ディレクトリ
if not os.path.exists(output_dir): os.mkdir(output_dir) # ディレクトリ生成

# 作成したカスタム・データセット
train_dataset = ld.ImageFolderAnnotationRect(img_dir_path, annot_file_name, ld.get_transform(train=True))
val_dataset = ld.ImageFolderAnnotationRect(img_dir_path, annot_file_name, ld.get_transform(train=False))

# データセットを訓練セットとテストセットに分割
# torch.manual_seed(1)
indices = torch.randperm(len(train_dataset)).tolist()
train_data_size = int(cf.splitRateTrain * len(indices))
# train_dataset = torch.utils.data.Subset(train_dataset, indices[:train_data_size])
val_dataset = torch.utils.data.Subset(val_dataset, indices[train_data_size:])
print(len(indices), len(train_dataset), len(val_dataset))

# 訓練データと評価データのデータロード用オブジェクトを用意
train_loader = DataLoader(train_dataset, batch_size=cf.batchSize, shuffle=False, num_workers=int(os.cpu_count() / 2), collate_fn=ld.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=int(os.cpu_count() / 2), collate_fn=ld.collate_fn)

# モデル、損失関数、最適化関数、収束率の定義
model = cf.build_model().to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) # 3エポックごとに学習率が1/10　５回目あたりまでの伸び率が大きかったらstepの値を大きくしてもいいかも

with open(path_log, mode = "w") as f: f.write("")
s_tm = time.time()
for epoch in range(cf.epochSize):
    loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, print_freq=1) # 学習
    lr_scheduler.step() # 学習率の更新 45行目コメントアウトすると39行目をやらないことになる
    f1v = evaluate(model, val_loader, device=DEVICE) # テストデータセットの評価
    torch.save(model.state_dict(), f"{output_dir}/_m_{id_str}_{epoch + 1:03}.pth") # モデルの保存

    # 学習の状況をCSVに保存
    with open(path_log, mode = "a") as f: f.write(f"{loss},{f1v}\n")
print("done %.0fs" % (time.time() - s_tm))