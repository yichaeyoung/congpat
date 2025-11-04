import os
import torch
from torch.utils.data import DataLoader

# custom
from utils.summarizer.s_dataloader_5 import summarizer_dataloader, pad_batch
from utils.summarizer.s_train import train
from architectures.summarizer.lstm_v1 import LSTMAutoEncoder


def get_device():

    gpu_number = 0
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon (MPS)"
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_number}")
        # cuda:0 --> screen gpu_0
        # cuda:1 --> screen gpu_1
        # cuda --> screen training


        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        print('Error Code 0 : No GPU, No Training Go Home')
        sys.exit()
    
    print(f"Device Name : {device_name}")

    return device


def main():

    file_names = [
        'apsiii', # 0
        'bg', # 1
        'chemistry', # 2 
        'complete_blood_count', # 3
        'creatinine_baseline', # 4
        'crrt', # 5
        'enzyme', # 6
        'inflammation' , # 7 
        'kdigo_creatinine' , # 8 
        'kdigo_stages', # 9
    ]
    # --------------------
    # 0) 기본 설정
    # --------------------
    epochs = 500
    lr = 1e-3
    grad_clip = 1.0

    file_name = file_names[8]
    reference_file_name = 'labevents' #'icustays_filtered_jh'
    device = get_device()
    logdir = f"./runs/summarizer_{file_name}_{epochs}epoch_{reference_file_name}rf"          # TensorBoard 로그 경로
    os.makedirs(logdir, exist_ok=True)

    # --------------------
    # 1) Dataset & Loader
    # --------------------
    dataset = summarizer_dataloader(
        column_key_name="hadm_id",
        master_csv_path = f'/home/jihoney/workdir/assist_workdir/coddyddld_workspace/jihoney_space/dataset_building/reference_csv/{reference_file_name}.csv',
        derived_csv_path = f'/home/jihoney/workdir/assist_workdir/coddyddld_workspace/jihoney_space/dataset_building/outputs/{file_name}_filtered.csv',
        # 필요 시 여러분의 CSV 경로 등 옵션 인자들 추가
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,          # subject_id 순서 유지 시 False 권장
        num_workers=4,          # IO 병렬 (CSV 크면 2~8 정도)
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=pad_batch,   # (B,T,F) 패딩
        drop_last=False,
    )


    sample = next(iter(loader))
    X_sample = sample["X"]                # (B, T, F)
    feature_dim = X_sample.shape[-1]
    print(f"[Info] feature_dim = {feature_dim}")


    model = LSTMAutoEncoder(
        input_dim=feature_dim,
        hidden_dim=128,
        latent_dim=64,
        num_layers=2,
        bidirectional=False,
        dropout=0.1,
    ).to(device)


    
    train(
        model=model,
        train_loader=loader,
        epochs=epochs,
        device=device,
        tb_logdir=logdir,
        lr=lr,
        grad_clip=grad_clip,
    )

    torch.save(model.state_dict(), f"./trained_models/summarizer/summarizer_{file_name}_{epochs}epoch.pth")

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        X = batch["X"].to(device)              # (B,T,F)
        lengths = batch["lengths"].to(device)  # (B,)
        X_hat, z = model(X, lengths=lengths, return_latent=True)
        print(f"[Info] X_hat shape: {X_hat.shape}, latent z shape: {z.shape}")

if __name__ == "__main__":
    main()