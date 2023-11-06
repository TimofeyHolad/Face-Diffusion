
import torch
import torch.nn as nn 


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, img_size, num_heads=8, activation=nn.GELU):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        
        self.mha = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm([img_size * img_size, in_channels])
        self.ff = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=4*in_channels),
            activation(),
            nn.Linear(in_features=4*in_channels, out_features=in_channels)
        )
        self.norm2 = nn.LayerNorm([img_size * img_size, in_channels])
        
    def forward(self, x):
        x = x.view(-1, self.in_channels, self.img_size*self.img_size).swapaxes(1, 2)
        mha_out = self.mha(x, x, x, need_weights=False, average_attn_weights=False)[0]
        x = x + mha_out
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm2(x)
        x = x.swapaxes(1, 2).view(-1, self.in_channels, self.img_size, self.img_size)
        return x
    
class TimeEncoder(nn.Module):
    def __init__(self, timestamps_num, time_embeddings_size, n=10000):
        super().__init__()
        time_embeddings_size = (time_embeddings_size // 2) * 2
        k = torch.arange(timestamps_num)[:, None]
        i = torch.arange(time_embeddings_size / 2)[None, :]
        
        x = k / (n ** (2 * i / time_embeddings_size))
        self.embeddings = torch.zeros(size=(timestamps_num, time_embeddings_size), requires_grad=False)
        self.embeddings[:, 0::2] = torch.sin(x)
        self.embeddings[:, 1::2] = torch.cos(x)
        
    def forward(self, t):
        if isinstance(t, int):
            t = torch.tensor(t)[None]
        embeddings = self.embeddings[t]
        return embeddings
    
class DoubleConv2d(nn.Module):
    def __init__(
        self, 
        in_channels, 
        mid_channels, 
        out_channels, 
        time_embeddings_size, 
        time_activation=nn.SiLU, 
        activation=nn.GELU, 
        concat_input=False, 
        residual_connection=False
    ):
        super().__init__()
        self.concat_input = concat_input
        self.residual_connection = residual_connection
        
        if concat_input:
            in_channels *= 2
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.act1 = activation()
        
        self.time_linear = nn.Linear(in_features=time_embeddings_size, out_features=mid_channels)
        self.time_act = time_activation()
        
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = activation()
        
        if residual_connection:
            self.residual_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        
        
    def forward(self, x, t, concat_tensor=None):
        if self.concat_input:
            x = torch.cat(tensors=(x, concat_tensor), dim=1)
        add = 0
        if self.residual_connection:
            add = self.residual_conv(x)
        x = self.act1(self.norm1(self.conv1(x)))
        t = self.time_act(self.time_linear(t))
        x = x + t[:, :, None, None]
        x = self.conv2(x) + add
        x = self.act2(self.norm2(x))
        return x
    
class DownBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        time_embeddings_size, 
        time_activation=nn.SiLU, 
        activation=nn.GELU, 
        residual_connection=False
    ):
        super().__init__()
        
        self.down_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1)
        self.down_norm = nn.BatchNorm2d(in_channels)
        self.down_act = activation()
        
        self.double_conv = DoubleConv2d(
            in_channels=in_channels, 
            mid_channels=out_channels, 
            out_channels=out_channels, 
            time_embeddings_size=time_embeddings_size, 
            time_activation=time_activation, 
            activation=activation, 
            concat_input=False, 
            residual_connection=residual_connection
        )
        
    
    def forward(self, x, t):
        x = self.down_act(self.down_norm(self.down_conv(x)))
        x = self.double_conv(x, t)
        return x
    
class UpBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        time_embeddings_size, 
        time_activation=nn.SiLU, 
        activation=nn.GELU, 
        concat_input=False, 
        residual_connection=False
    ):
        super().__init__()
        self.residual_conection = concat_input
        
        self.double_conv = DoubleConv2d(
            in_channels=in_channels, 
            mid_channels=in_channels, 
            out_channels=in_channels, 
            time_embeddings_size=time_embeddings_size, 
            time_activation=time_activation, 
            activation=activation, 
            concat_input=concat_input, 
            residual_connection=residual_connection
        )
        
        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        self.up_norm = nn.BatchNorm2d(out_channels)
        self.up_act = activation()
        

    def forward(self, x, t, concat_tensor=None):
        x = self.double_conv(x, t, concat_tensor)
        x = self.up_act(self.up_norm(self.up_conv(x)))
        return x
    
class Unet(nn.Module):
    def __init__(
        self, 
        img_size, 
        model_size, 
        timestamps_num, 
        in_channels, 
        out_channels, 
        base_channels=32, 
        time_embeddings_size=32, 
        time_activation=nn.SiLU, 
        activation=nn.GELU, 
        residual_connection=True
    ):
        assert img_size // (2 ** model_size) >= 1, 'model_size is too big'
        super().__init__()
        self.img_size             = img_size
        self.model_size           = model_size
        self.in_channels          = in_channels
        self.out_channels         = out_channels
        self.base_channels        = base_channels
        self.timestamps_num       = timestamps_num
        self.time_embeddings_size = time_embeddings_size
        
        self.time_encoder = TimeEncoder(timestamps_num=timestamps_num, time_embeddings_size=time_embeddings_size)
        
        self.entry_conv = DoubleConv2d(
            in_channels=in_channels, 
            mid_channels=base_channels, 
            out_channels=base_channels, 
            time_embeddings_size=time_embeddings_size, 
            time_activation=time_activation, 
            activation=activation, 
            concat_input=False, 
            residual_connection=residual_connection
        )
        self.downs_attentions = nn.ModuleList([AttentionBlock(in_channels=base_channels * (2 ** i), img_size=int(img_size / (2 ** i)))
                                               for i in range(model_size - 1)])
        self.downs = nn.ModuleList([DownBlock(
                                        in_channels=base_channels * (2 ** i), 
                                        out_channels=base_channels * (2 ** (i + 1)), 
                                        time_embeddings_size=time_embeddings_size, 
                                        time_activation=time_activation,
                                        activation=activation,
                                        residual_connection=residual_connection
                                    ) 
                                    for i in range(model_size - 1)])
        
        self.entry_bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=base_channels * (2 ** (model_size - 1)), 
                out_channels=base_channels * (2 ** (model_size - 1)), 
                kernel_size=4, 
                stride=2, 
                padding=1
            ),
            nn.BatchNorm2d(base_channels * (2 ** (model_size - 1))),
            activation()
        )
        self.bottleneck = DoubleConv2d(
            in_channels=base_channels * (2 ** (model_size - 1)), 
            mid_channels=base_channels * (2 ** model_size), 
            out_channels=base_channels * (2 ** (model_size - 1)), 
            time_embeddings_size=time_embeddings_size, 
            time_activation=time_activation,
            activation=activation, 
            concat_input=False,
            residual_connection=residual_connection
        )
        self.exit_bottleneck = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=base_channels * (2 ** (model_size - 1)), 
                out_channels=base_channels * (2 ** (model_size - 1)), 
                kernel_size=4, 
                stride=2, 
                padding=1
            ),
            nn.BatchNorm2d(base_channels * (2 ** (model_size - 1))),
            activation()
        )
        
        self.ups = nn.ModuleList([UpBlock(
                                      in_channels=base_channels * (2 ** i), 
                                      out_channels=base_channels * (2 ** (i - 1)), 
                                      time_embeddings_size=time_embeddings_size, 
                                      time_activation=time_activation,
                                      activation=activation,
                                      concat_input=True, 
                                      residual_connection=residual_connection
                                  ) 
                                  for i in range(model_size - 1, 0, -1)])
        self.ups_attentions = nn.ModuleList([AttentionBlock(in_channels=base_channels * (2 ** (i - 1)), img_size=int(img_size / (2 ** (i - 1))))
                                             for i in range(model_size - 1, 0, -1)])
        self.exit_conv = DoubleConv2d(
            in_channels=base_channels, 
            mid_channels=base_channels, 
            out_channels=base_channels, 
            time_embeddings_size=32, 
            time_activation=time_activation, 
            activation=activation, 
            concat_input=True, 
            residual_connection=residual_connection
        )
        self.out_conv = nn.Conv2d(in_channels=base_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x, t):
        t = self.time_encoder(t)
        t = t.to(x.device, dtype=x.dtype)
        x = self.entry_conv(x, t)
        concat_tensors = [x,]
        for i in range(len(self.downs)):
            x = self.downs_attentions[i](x)
            x = self.downs[i](x, t)
            concat_tensors.append(x)
        x = self.entry_bottleneck(x)
        x = self.bottleneck(x, t)
        x = self.exit_bottleneck(x)
        for i in range(len(self.ups)):
            concat_tensor = concat_tensors.pop()
            x = self.ups[i](x, t, concat_tensor)
            x = self.ups_attentions[i](x)
        concat_tensor = concat_tensors.pop()
        x = self.exit_conv(x, t, concat_tensor)
        x = self.out_conv(x)
        return x