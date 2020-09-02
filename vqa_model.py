# coding=utf-8
# Copyleft 2019 project LXRT.
# copied from LXRT with modifications
import torch.nn as nn

from param import args
from src.entry import LXRTEncoder, VBEncoder, UniterEncoder
from src.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = args.max_seq_length


class VQAModel(nn.Module):
    def __init__(self, num_answers, model = 'lxmert'):
        super().__init__()
        self.model = model
        # Build LXRT encoder
        if model == 'lxmert':
            self.encoder = LXRTEncoder(args)
            
        elif model == 'visualbert':
            self.encoder = VBEncoder(args)
        elif model == 'uniter':
            self.encoder = UniterEncoder(args)
        
        hid_dim = self.encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        if self.model == 'lxmert':
            x = self.encoder(sent, (feat, pos))
        elif self.model == 'visualbert':
            x = self.encoder(sent, feat)
        elif self.model == 'uniter':
            x = self.encoder(sent, feat, pos)
        logit = self.logit_fc(x)

        return logit
    


