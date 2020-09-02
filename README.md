# Transformers-VQA
An implementation of down-streaming trending pre-trained V+L models to VQA tasks. 
Now support: VisualBERT, LXMERT, and UNITER.

### USAGE

Following is an example of fine-tuning VQA 2.0 dataset.

**0.0**. Clone our repo.

**0.1**. Install all python dependencies (a virtual environment is highly recommended):
```sh
pip install -r requirements.txt
```

**0.2**. Download pre-trained models and place them to data/pretrained/

You can download these models from their own github repo. We also provide command lines to handle this:

VisualBERT:
```sh
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kuPr187zWxSJbtCbVW87XzInXltM-i9Y' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kuPr187zWxSJbtCbVW87XzInXltM-i9Y" -O models/pretrained/visualbert.th && rm -rf /tmp/cookies.txt
```
UNITER:
```sh
wget https://convaisharables.blob.core.windows.net/uniter/pretrained/uniter-base.pt -P models/pretrained/uniter-base.th
```
LXMERT:
```sh
wget --no-check-certificate https://nlp1.cs.unc.edu/data/model_LXRT.pth -P models/pretrained/model_LXRT.pth


