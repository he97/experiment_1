from models.Trans_BCDM_A.net_A import *
import torch

nClass = 7
dim = 512
in_chans = 48
patch_dim = 512
a = DTransformer(
    in_chans = in_chans,
    num_patches=dim,
    patch_dim = patch_dim,
    image_size=5,
    patch_size=5,
    attn_layers=Encoder(
        dim=512,
        depth=2,
        heads=2))
b = torch.randn(32, 48, 5, 5)
c = a(b)
print(a(b))
