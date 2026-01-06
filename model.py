import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encodes the time 't' into a vector. 
    Neural networks cannot handle a single float number well. 
    We convert t=0.5 into a vector like [0.1, -0.5, 0.9, ...] using sine/cosine.
    This is standard in Transformers and Diffusion models.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """
    A basic Residual Block: Conv -> GroupNorm -> SiLU -> Conv.
    It processes information while keeping the gradient flow stable (ResNet).
    """
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act1 = nn.SiLU()
        
        # Time integration: We shift/scale the features based on time
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2)
        )
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()

    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        
        # Inject Time Info
        t_emb = self.mlp(time_emb)
        # Reshape to broadcast over spatial dimensions
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        scale, shift = t_emb.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        return h + x if x.shape[1] == h.shape[1] else h

class FlowMatchingNet(nn.Module):
    """
    A simplified U-Net for Flow Matching.
    Input:
        x: Noisy Image (Batch, 3, 64, 64)
        t: Time (Batch,)
        cond: Landmark Image (Batch, 3, 64, 64) - Acts as the guide
    Output:
        velocity: (Batch, 3, 64, 64) - The direction to move pixels
    """
    def __init__(self, image_channels=3):
        super().__init__()
        
        # Hyperparameters
        down_channels = [64, 128, 256]
        time_emb_dim = 32

        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # Initial Projection
        # We input 6 channels: 3 for noisy image + 3 for landmark condition
        self.conv_in = nn.Conv2d(image_channels * 2, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([])
        for i in range(len(down_channels)-1):
            self.downs.append(nn.ModuleList([
                Block(down_channels[i], down_channels[i], time_emb_dim),
                nn.Conv2d(down_channels[i], down_channels[i+1], 4, 2, 1) # Stride 2 to downsample
            ]))

        # Bottleneck (Middle)
        self.mid_block1 = Block(down_channels[-1], down_channels[-1], time_emb_dim)
        self.mid_block2 = Block(down_channels[-1], down_channels[-1], time_emb_dim)

        # Upsample
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(down_channels)-1)):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(down_channels[i+1], down_channels[i], 4, 2, 1), # Upsample
                Block(down_channels[i] * 2, down_channels[i], time_emb_dim) # *2 because we concat skip connections
            ]))

        # Final Output
        self.conv_out = nn.Conv2d(down_channels[0], image_channels, 3, padding=1)

    def forward(self, x, t, cond):
        # 1. Embed Time
        t_emb = self.time_mlp(t)
        
        # 2. Concatenate Noisy Image + Landmark Condition
        # x shape: (B, 3, 64, 64)
        # cond shape: (B, 3, 64, 64)
        # x_in shape: (B, 6, 64, 64)
        x_in = torch.cat([x, cond], dim=1)
        
        # 3. Initial Conv
        h = self.conv_in(x_in)
        
        # 4. Downsample (Encoder)
        skips = []
        for block, downsample in self.downs:
            h = block(h, t_emb)
            skips.append(h)
            h = downsample(h)
            
        # 5. Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)
        
        # 6. Upsample (Decoder)
        for upsample, block in self.ups:
            h = upsample(h)
            # Skip connection: add features from the encoder path
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
            
        # 7. Final Velocity Prediction
        return self.conv_out(h)

if __name__ == "__main__":
    # Test the model shapes
    print("Testing Model Architecture...")
    net = FlowMatchingNet()
    
    # Dummy data
    B = 2
    x = torch.randn(B, 3, 64, 64)       # Noisy image
    cond = torch.randn(B, 3, 64, 64)    # Landmark mask
    t = torch.rand(B)                   # Time
    
    output = net(x, t, cond)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Success! The output shape matches the input, representing velocity vector field.")
