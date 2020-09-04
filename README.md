# Transformers-VQA
An implementation of down-streaming trending pre-trained V+L models to VQA tasks. 
Now support: VisualBERT, LXMERT, and UNITER.

### USAGE

Following is an example of fine-tuning VQA 2.0 dataset.

**0**. Clone our repo.

**1**. Install all python dependencies (a virtual environment is highly recommended):
```sh
pip install -r requirements.txt
```

**2**. Download pre-trained models and place them to data/pretrained/

You can download these models from their own github repo. We also provide command lines to handle this:

  **VisualBERT:**
```sh
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kuPr187zWxSJbtCbVW87XzInXltM-i9Y' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kuPr187zWxSJbtCbVW87XzInXltM-i9Y" -O models/pretrained/visualbert.th && rm -rf /tmp/cookies.txt
```
  **UNITER:**
```sh
wget https://convaisharables.blob.core.windows.net/uniter/pretrained/uniter-base.pt -P models/pretrained/
```
  **LXMERT:**
```sh
wget --no-check-certificate https://nlp1.cs.unc.edu/data/model_LXRT.pth -P models/pretrained/
```

**3** Download re-distributed json files for VQA 2.0 (copy from airsplay/lxmert)
```sh
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/vqa/train.json -P data/
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/vqa/nominival.json -P  data/
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/vqa/minival.json -P data/
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/vqa/test.json -P data/
```
**4** Download faster-rcnn features for MS COCO train2014 (17 GB) and val2014 (8 GB) images (copy from airsplay/lxmert), this process will take a while
```sh
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/img
unzip data/img/train2014_obj36.zip -d data/img && rm data/img/train2014_obj36.zip
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/img
unzip data/img/val2014_obj36.zip -d data && rm data/img/val2014_obj36.zip
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/test2015_obj36.zip -P data/img
unzip data/img/test2015_obj36.zip -d data && rm data/img/test2015_obj36.zip
```
**5** Now you have fulfill all requirements and dependencies, run this command before fine-tuning on the entire training dataset:
```sh
python vqa.py --tiny
```
**6** If no error pops up, you are good to go. Please refer param.py for all settings. Here is an example of fine-tuning UNITER:
```sh
python vqa.py --model uniter --epochs 6 --max_seq_length 20 --load_pretrained models/pretrained/uniter-base.pt --output models/trained/
```
**7** 

