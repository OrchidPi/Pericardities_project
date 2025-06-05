import torch
import torch.nn as nn
import torch.nn.functional as F
# from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from MedCLIP.medclip.modeling_medclip import MedCLIPModel, PromptClassifier, MedCLIPVisionModel, MedCLIPVisionModelViT


class VIT(nn.Module):
    def __init__(self, cfg, num_labels=4, feature_dim=512):
        super(VIT, self).__init__()
        self.cfg = cfg
        #self._init_classifier()
        #self._init_bn()

        # Use convnext_base as the backbone, but exclude the head
        self.backbone = MedCLIPVisionModelViT(checkpoint=None)
        # Confounder classifier
        self.ecg = nn.Sequential(
            nn.Linear(feature_dim, int(feature_dim/4)),
            nn.ReLU()
        )

        self.conf_classifiers = nn.ModuleList()
        for idx, num_class in enumerate(self.cfg.num_conf):
            # Create a separate linear layer for each task
            fc = nn.Linear(int(feature_dim/4), num_class)
            self.conf_classifiers.append(fc)

        # Define one fully connected layer per binary label
        self.classifier = nn.Linear(feature_dim, 1)

   
    def forward(self, x):
        # Extract features from backbone
        x, ECG_early_layer = self.backbone(x, return_layer=3)

        ECG_early_layer = self.ecg(ECG_early_layer)  # [N, num_patches, feature_dim//4]
        ECG_early_layer = ECG_early_layer.mean(dim=1)  # [N, feature_dim//4]

        # Main classification head
        x = F.dropout(x, p=self.cfg.fc_drop, training=self.training)
        logit = self.classifier(x)  # [N, 1]

        # Confounder classification heads
        conf_logits = []
        for conf_classifier in self.conf_classifiers:
            conf_logits.append(conf_classifier(ECG_early_layer))  # e.g. [N, num_conf_classes]

        return logit, conf_logits