ofc have ollama installed on pc
need tesseract

```bash
sudo apt-get install tesseract-ocr
```

## venv

i need to use 3.12

```bash
uv python pin 3.12
uv venv
```

## if nvidia

i think its just
`uv pip install torch torchvision`
(need to test)

## if amd

note im on debian 13 and doing `rocminfo` gives me runtime version 1.18 so i will use python 3.12 and install rocm6.2

first uninstall current torch and install
`uv pip uninstall torch torchvision`
then install romc version
`uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2`
verify gpu detected
`python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"`
or

```bash
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('HIP version:', torch.version.hip)
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
"
```
