import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.resnet import Bottleneck

BODY_PARTS = [
    "head", "neck", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_hand", "right_hand",
    "left_foot", "right_foot",
]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_INPUT_SIZE    = 256


class _CBR(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _PartBranch(nn.Module):
    def __init__(self, num_parts=18):
        super().__init__()
        self.num_parts = num_parts
        self.cbr = nn.ModuleList([_CBR(512, 64) for _ in range(num_parts)])
        self.conv_last_part = nn.Conv2d(512, num_parts, 1)
        self.conv_last_cont = nn.Conv2d(num_parts * 64, num_parts, 1)

    def forward(self, part_feat):
        per_part = [self.cbr[i](part_feat) for i in range(self.num_parts)]
        concat   = torch.cat(per_part, dim=1)           # [B, 18*64, H, W]
        part_seg  = self.conv_last_part(part_feat)       # [B, 18, H, W]
        part_cont = self.conv_last_cont(concat)          # [B, 18, H, W]
        return part_seg, part_cont


class _HOTEncoder(nn.Module):
    """ResNet-50 with 3-layer deep stem (conv1→conv2→conv3 before layers)."""

    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3,   64,  3, stride=2, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.conv2   = nn.Conv2d(64,  64,  3, stride=1, padding=1, bias=False)
        self.bn2     = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64,  128, 3, stride=1, padding=1, bias=False)
        self.bn3     = nn.BatchNorm2d(128)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1  = self._make_layer(128,  64, 3, stride=1)
        self.layer2  = self._make_layer(256, 128, 4, stride=2)
        self.layer3  = self._make_layer(512, 256, 6, stride=2)
        self.layer4  = self._make_layer(1024, 512, 3, stride=2)

    def _make_layer(self, inplanes, planes, blocks, stride):
        downsample = None
        if stride != 1 or inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        layers = [Bottleneck(inplanes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(Bottleneck(planes * 4, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class _HOTDecoder(nn.Module):
    def __init__(self, num_parts=18):
        super().__init__()
        self.cbr          = _CBR(2048, 512)
        self.cbr_part     = _CBR(2048, 512)
        self.part_branch  = _PartBranch(num_parts)
        self.conv_last    = nn.Conv2d(512, num_parts, 1)
        self.conv_binary  = nn.Conv2d(512, 2, 1)

    def forward(self, feat):
        main_feat          = self.cbr(feat)
        part_feat          = self.cbr_part(feat)
        part_seg, part_cont = self.part_branch(part_feat)
        contact_map        = self.conv_last(main_feat)
        binary             = self.conv_binary(main_feat)
        return binary, contact_map, part_seg, part_cont


class HOTDetector:
    """
    Wraps the HOT encoder-decoder for per-frame contact inference.
    Exposes: infer(frame_bgr, box) -> (contact_prob: float, top_parts: list[str])
    """

    ENC_PATH = "models/hot-c1/encoder_epoch_14.pth"
    DEC_PATH = "models/hot-c1/decoder_epoch_14.pth"

    def __init__(self, enc_path=ENC_PATH, dec_path=DEC_PATH, device=None):
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = _HOTEncoder().to(self.device)
        self.decoder = _HOTDecoder().to(self.device)

        self.encoder.load_state_dict(
            torch.load(enc_path, map_location=self.device), strict=False
        )
        self.decoder.load_state_dict(
            torch.load(dec_path, map_location=self.device), strict=False
        )
        self.encoder.eval()
        self.decoder.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((_INPUT_SIZE, _INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        print("HOT detector loaded.")

    @torch.no_grad()
    def infer(self, frame_bgr, box):
        """
        Args:
            frame_bgr : full frame as numpy BGR array
            box       : [x1, y1, x2, y2]
        Returns:
            contact_prob : float  (0–1, probability of contact)
            top_parts    : list[str]  top-3 body parts in contact
        """
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]

        # Pad crop by 20% of box size
        pad = int(max(x2 - x1, y2 - y1) * 0.2)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0, []

        rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        feat                              = self.encoder(tensor)
        binary, contact_map, _, _        = self.decoder(feat)

        # contact probability: softmax over binary map, mean across spatial
        contact_prob = float(torch.softmax(binary, dim=1)[0, 1].mean())

        # top-3 body parts by average activation
        part_scores = contact_map[0].mean(dim=(1, 2))
        top_idx     = part_scores.argsort(descending=True)[:3].cpu().tolist()
        top_parts   = [BODY_PARTS[i] for i in top_idx if i < len(BODY_PARTS)]

        return contact_prob, top_parts
