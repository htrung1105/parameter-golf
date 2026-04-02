# 🚀 Hướng dẫn chạy test trên Remote GPU

## Bước 1: Cấu hình Container khi thuê

| Setting | Giá trị |
|---|---|
| **GPU** | 1× RTX 4090 (54 cores, 135GB RAM) |
| **Framework** | PyTorch 2.9.1 (CUDA 12.8) |
| **CPU** | 8 cores |
| **RAM** | 32 GB |
| **Storage** | 24 GB |
| **SSH Password** | Đặt mật khẩu (ví dụ: `PgOlf2026!Xsa`) |
| **SSH Username** | Để trống (mặc định `root`) |
| **SSH Public Key** | Dán nội dung bên dưới ⬇️ |
| **Environment Variables** | Để trống |
| **Exposed Ports** | Để trống |

### Public key của bạn (dán vào ô SSH Public Key):

```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDKi2ERexP... (xem file ~/.ssh/id_rsa.pub)
```

> Chạy lệnh sau để copy: `cat $env:USERPROFILE\.ssh\id_rsa.pub | clip`

---

## Bước 2: Kết nối SSH

Sau khi container khởi động, bạn sẽ nhận được thông tin:
- **Host**: ví dụ `gpu-node-03.example.com`  
- **Port**: ví dụ `22` hoặc `2222`

### Từ PowerShell (Windows Terminal):

```powershell
ssh root@HOST -p PORT
# Nhập password khi được hỏi
```

---

## Bước 3: Setup trên Remote (chạy 1 lần)

Sau khi SSH vào, chạy từng bước:

```bash
# Fix pip
export PATH=/opt/conda/bin:$PATH

# Di chuyển vào workspace
cd /workspace

# Clone repo gốc
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Cài dependencies
pip install sentencepiece numpy huggingface-hub datasets tqdm

# Tải dataset (1 shard cho test, ~200MB + 124MB val)
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# Kiểm tra GPU
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

---

## Bước 4: Upload code từ local

### Cách 1 — SCP (từ PowerShell trên Windows):

```powershell
# Upload train_gpt.py lên remote (thay HOST và PORT)
scp -P PORT d:\Y3S2\parameter-golf\train_gpt.py root@HOST:/workspace/parameter-golf/train_gpt.py

# Upload cả 2 script tiện ích
scp -P PORT d:\Y3S2\parameter-golf\runpod_1gpu.sh root@HOST:/workspace/parameter-golf/
scp -P PORT d:\Y3S2\parameter-golf\runpod_8gpu.sh root@HOST:/workspace/parameter-golf/
```

### Cách 2 — Copy-paste qua terminal:

```bash
# Trên remote, tạo file và paste nội dung
cat > /workspace/parameter-golf/train_gpt.py << 'ENDOFFILE'
# ... (paste toàn bộ nội dung train_gpt.py) ...
ENDOFFILE
```

---

## Bước 5: Chạy Test Training

```bash
cd /workspace/parameter-golf

# Cấp quyền thực thi
chmod +x runpod_1gpu.sh

# Chạy training (~15-20 phút trên 1x RTX 4090)
bash runpod_1gpu.sh
```

### Hoặc chạy trực tiếp (không dùng script):

```bash
cd /workspace/parameter-golf

SEED=42 ITERATIONS=3000 MAX_WALLCLOCK_SECONDS=420 \
TRAIN_BATCH_TOKENS=98304 NUM_LAYERS=11 MODEL_DIM=512 \
BIGRAM_VOCAB_SIZE=4096 TRIGRAM_VOCAB_SIZE=4096 \
XSA_LAST_N=4 GATED_ATTENTION=1 VALUE_RESIDUAL=1 \
DEPTH_RECUR_PASSES=2 LATE_QAT_THRESHOLD=0.15 \
SWA_ENABLED=1 TORCH_COMPILE=1 TTT_ENABLED=0 \
TRAIN_LOG_EVERY=100 VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

---

## Bước 6: Kiểm tra kết quả

Khi training xong, bạn sẽ thấy output tương tự:

```
step:3000/3000 loss:X.XXXX t:XXXXXms avg:XX.XXms
peak_mem:XXXXMiB
ema:applying
post_ema val_loss:X.XXXX val_bpb:X.XXXX
int6_roundtrip val_loss:X.XXXX val_bpb:X.XXXX
int8_zlib code:XXXXX model:XXXXXXX total:XXXXXXX limit:16000000 OK  ← ✅
sliding val_loss:X.XXXX val_bpb:X.XXXX
```

### Các chỉ số quan trọng cần kiểm tra:

| Chỉ số | Mong đợi (1×4090, 3K steps) | Ý nghĩa |
|---|---|---|
| `train_loss` cuối | ~3.5-4.5 | Loss đang giảm = code đúng |
| `val_bpb` | ~1.5-2.0 | Chưa tối ưu (cần 8×H100 full run) |
| `total` bytes | < 16,000,000 | ✅ Artifact fits 16MB limit |
| `int6_roundtrip` | Gần `post_ema` | Quantization không hỏng model |
| `peak_mem` | < 20,000 MiB | Vừa VRAM của 4090 (24GB) |

---

## Bước 7: Download kết quả về local

```powershell
# Từ PowerShell trên Windows (thay HOST và PORT)
scp -P PORT root@HOST:/workspace/parameter-golf/final_model.int8.ptz d:\Y3S2\parameter-golf\
scp -P PORT root@HOST:/workspace/parameter-golf/logs/*.txt d:\Y3S2\parameter-golf\logs\
```

---

## ⚠️ Lưu ý quan trọng

1. **Đây là test run, KHÔNG phải submission** — BPB trên 1×4090 với 3K steps sẽ cao hơn nhiều so với production (1.12 target cần 8×H100 full 10 phút)
2. **Mục đích**: Xác nhận code chạy end-to-end, serialization đúng, artifact ≤ 16MB
3. **Sau khi test thành công**: Thuê 8×H100 SXM trên RunPod để chạy submission thật
4. **Nhớ tắt container** sau khi xong để không bị charge thêm!

---

## Tổng hợp

```bash
# PowerShell
ssh root@HOST -p PORT

scp -P PORT .\remote_setup.sh root@HOST:/workspace/remote_setup.sh
```

```bash
# Remote
export PATH=/opt/conda/bin:$PATH

tmux new -s train

cd /workspace

chmod +x remote_setup.sh

# Usage: bash remote_setup.sh [num_shards]
bash remote_setup.sh 1    # Quick test (1 shard, ~200MB)
bash remote_setup.sh 80   # Full dataset (80 shards, ~16GB)
```

```bash
# PowerShell
scp -P PORT .\train_gpt.py root@HOST:/workspace/parameter-golf/train_gpt.py

scp -P PORT .\runpod_1gpu.sh root@HOST:/workspace/parameter-golf/runpod_1gpu.sh

scp -P PORT .\runpod_2gpu.sh root@HOST:/workspace/parameter-golf/runpod_2gpu.sh

scp -P PORT .\runpod_8gpu.sh root@HOST:/workspace/parameter-golf/runpod_8gpu.sh
```

```bash
# Remote
cd parameter-golf

chmod +x runpod_2gpu.sh

bash runpod_2gpu.sh

# If disconnect
tmux attach -t train
```