import torch
import torch.nn as nn

from transformers import BertModel, BertConfig
from transformers import RobertaModel, RobertaConfig
import os
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import numpy as np
from PIL import Image

from MCA_Fusion import MCA


class BaseModel(nn.Module):
    def __init__(self, save_dir):
        super(BaseModel, self).__init__()
        self.save_dir = save_dir

    def save(self, filename):
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(self.save_dir, filename + '.pt'))

    def load(self, filepath):
        state_dict = torch.load(filepath)
        self.load_state_dict(state_dict, strict=False)


class MMModel(BaseModel):
    def __init__(self, imageEncoder, textEncoder, save_dir):
        super(MMModel, self).__init__(save_dir=save_dir)
        self.imageEncoder = imageEncoder
        self.textEncoder = textEncoder

    def forward(self, x):
        raise NotImplemented


class TextOnlyModel(BaseModel):
    def __init__(self, save_dir, dim_text_repr=768, num_class=2):
        super(TextOnlyModel, self).__init__(save_dir)
        config = BertConfig()
        self.textEncoder = BertModel(config).from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(dim_text_repr, num_class)

    def forward(self, x):
        _, text = x
        hidden_states = self.textEncoder(**text)  # B, T, dim_text_repr
        e_i = F.dropout(hidden_states[1])  # N, dim_text_repr
        return self.linear(e_i)


class ImageOnlyModel(BaseModel):
    def __init__(self, save_dir, dim_visual_repr=1000, num_class=2):
        super(ImageOnlyModel, self).__init__(save_dir=save_dir)

       # self.imageEncoder = torch.hub.load( 'pytorch/vision:v0.8.0', 'densenet201', pretrained=True)
        self.imageEncoder = torch.hub.load("pytorch/vision",  "efficientnet_v2_m" )
 
        self.flatten_vis = nn.Flatten()
        self.linear = nn.Linear(dim_visual_repr, num_class)

    def forward(self, x):
        image, _ = x
        f_i =(self.imageEncoder(image))
        f_i = self.linear(F.dropout(self.flatten_vis(f_i)))
        return (f_i)


class MCAModel(MMModel):
    def __init__(self, save_dir, dim_visual_repr=1000, dim_text_repr=768, dim_proj=1000, num_class=2):
        self.save_dir = save_dir

        self.dim_visual_repr = dim_visual_repr
        self.dim_text_repr = dim_text_repr

        imageEncoder = torch.hub.load('pytorch/vision', "efficientnet_v2_m", weights="IMAGENET1K_V1")
        #you can load your pretrained weights from your unimodal model to benfit from domain knowledge
        #imageEncoder.load_state_dict(torch.load('./output/task1/image/efficientnetV2/best_edited.pt'))
   
      
        config = BertConfig()
        textEncoder= BertModel(config).from_pretrained('bert-base-uncased')
        #you can load your pretrained weights from your unimodal model to benfit from domain knowledge
        #textEncoder.load_state_dict(torch.load('./output/task1/text/BERT/best_edited.pt'))



        super(MCAModel, self).__init__(imageEncoder, textEncoder, save_dir)

        # Flatten image features to 1D array
        self.flatten_vis = torch.nn.Flatten()

        # Linear layers used to project embeddings to fixed dimension (eqn. 3)
        self.proj_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.proj_text = nn.Linear(dim_text_repr, dim_proj)

        self.proj_visual_bn = nn.BatchNorm1d(dim_proj)
        self.proj_text_bn = nn.BatchNorm1d(dim_proj)
        
        
        #MCA-Fusion Module
        self.Fusion = MCA(channel=2,reduction=1)

        # Classification layer
        #self.cls_layer = nn.Linear(dim_proj, num_class)
        self.cls_layer = nn.Linear(2*dim_proj, num_class)

    def forward(self, x):
        image, text = x

        f_i =self.imageEncoder(image)
        f_i = F.dropout(self.flatten_vis(f_i))
        
        hidden_states = self.textEncoder(**text)  # N, T, dim_text_repr
        e_i = F.dropout(hidden_states[1])  # N, dim_text_repr

        f_i_tilde = F.relu(self.proj_visual_bn(
            self.proj_visual(f_i))).unsqueeze(1)  # N,1, dim_proj
        e_i_tilde = F.relu(self.proj_text_bn(
            self.proj_text(e_i))).unsqueeze(1)  # N,1, dim_proj

        joint_repr = torch.cat((f_i_tilde, e_i_tilde), dim=1).unsqueeze(3)  # N, 2*1, dim_proj,1
        
        joint_repr_Fused =self.Fusion(joint_repr)  # N, 2*1, dim_proj,1


        
        b, c, d,l = joint_repr_Fused.size()

        joint_repr_Fused_2dim=joint_repr_Fused.view(b, -1) 
        
        x=self.cls_layer(joint_repr_Fused_2dim)

        
        return x
        
        

        
   
