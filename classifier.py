import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path

# ---------------- Model Definition (same as training) ----------------
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden=None, drop=0.):
        super().__init__()
        hidden = hidden or int(dim*4)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class DropPath(nn.Module):
    def __init__(self, p=0.):
        super().__init__()
        self.p = p
    def forward(self, x):
        if self.p == 0. or not self.training: return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = mask.floor_()
        return x.div(keep) * mask

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=3, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1,2).reshape(B,N,C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)  # Correct!
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio), drop)  # Just MLP, no extra args
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
        
class TinyVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, num_classes=5,
                 dim=192, depth=4, heads=3, mlp_ratio=4., drop=0.1, attn_drop=0.1, drop_path=0.05):
        super().__init__()
        self.patch = PatchEmbedding(img_size, patch_size, in_ch, dim)
        n_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1,1,dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches+1, dim))
        self.pos_drop = nn.Dropout(drop)
        dpr = torch.linspace(0, drop_path, steps=depth).tolist()
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, mlp_ratio, drop, attn_drop, dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.head(x[:,0])
        return logits

# ---------------- Settings ----------------
IMG_SIZE = 224
PATCH_SIZE = 16
DIM = 192
DEPTH = 4
HEADS = 3
MLP_RATIO = 4.0
DROP = 0.1
ATTN_DROP = 0.1
DROP_PATH = 0.05
NUM_CLASSES = 5

CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "best_model.pth"  # Path to your trained checkpoint

# ---------------- Preprocessing ----------------
def preprocess_image(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.0)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)  # add batch dim

# ---------------- Load Model ----------------
@st.cache_resource(show_spinner=True)
def load_model(ckpt_path=CHECKPOINT_PATH):
    model = TinyVisionTransformer(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        mlp_ratio=MLP_RATIO,
        drop=DROP,
        attn_drop=ATTN_DROP,
        drop_path=DROP_PATH
    ).to(DEVICE)
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

model = load_model()

# ---------------- Streamlit UI ----------------
st.title("Diabetic Retinopathy Stage Prediction")
st.write("Upload a retinal fundus image and the model will predict the DR stage.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)


    # Preprocess & Predict
    input_tensor = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    st.write(f"**Predicted DR Stage:** {CLASS_NAMES[pred_idx]}")
    #st.write(f"**Confidence:** {confidence*100:.2f}%")
