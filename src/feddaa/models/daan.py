"""Day-aligned cross-attention network used in the manuscript."""

import torch
from torch import nn


class DAAN(nn.Module):
    def __init__(self,spec,past_dim,future_dim,history_steps=336):
        super().__init__()
        if history_steps<48 or history_steps%48: raise ValueError('Complete 48-step days required')
        self.days=history_steps//48
        width=spec.get('slot_width',24); heads=spec.get('attention_heads',4)
        if width<=0 or heads<=0 or width%heads: raise ValueError('Width must be divisible by heads')
        self.radius=spec.get('slot_radius',1)
        if not 0<=self.radius<24: raise ValueError('Radius must be between 0 and 23')
        self.history_embedding=nn.Sequential(nn.Linear(past_dim,width),nn.GELU(),nn.LayerNorm(width))
        self.query_embedding=nn.Sequential(nn.Linear(future_dim+2,width),nn.GELU(),nn.LayerNorm(width))
        self.day_embedding=nn.Embedding(self.days,width)
        self.offset_embedding=nn.Embedding(2*self.radius+1,width)
        self.attention=nn.MultiheadAttention(width,heads,dropout=0.,batch_first=True)
        self.context_norm=nn.LayerNorm(width)
        self.history_head=nn.Sequential(nn.Linear(2*width+3*past_dim+future_dim+2,width),
                                        nn.GELU(),nn.Linear(width,1))
        nn.init.zeros_(self.history_head[-1].weight)
        nn.init.zeros_(self.history_head[-1].bias)

    def aligned_tokens(self,past):
        batch,steps,channels=past.shape
        if steps!=self.days*48: raise ValueError('History length mismatch')
        daily=past.reshape(batch,self.days,48,channels)
        positions=torch.arange(48,device=past.device)[:,None]
        offsets=torch.arange(-self.radius,self.radius+1,device=past.device)[None,:]
        # No circular wrap between the end and start of a day. Mask out invalid neighbors.
        raw_indices=positions+offsets
        valid=(raw_indices>=0)&(raw_indices<48)
        indices=raw_indices.clamp(0,47)
        local=daily[:,:,indices,:].permute(0,2,1,3,4)
        tokens=self.history_embedding(local)
        day=self.day_embedding(torch.arange(self.days,device=past.device))[None,None,:,None,:]
        offset=self.offset_embedding(torch.arange(2*self.radius+1,device=past.device))[None,None,None,:,:]
        tokens=(tokens+day+offset).reshape(batch*48,self.days*(2*self.radius+1),-1)
        mask=(~valid[:,None,:].expand(48,self.days,-1)).reshape(48,-1)
        mask=mask[None,:,:].expand(batch,-1,-1).reshape(batch*48,-1)
        return daily,tokens,mask

    def forward_with_attention(self,past,future):
        if future.shape[1]!=48: raise ValueError('Forecast length mismatch')
        batch=past.shape[0]
        daily,tokens,mask=self.aligned_tokens(past)
        phase=torch.arange(48,device=past.device,dtype=past.dtype)*(2*torch.pi/48)
        clock=torch.stack([phase.sin(),phase.cos()],-1)[None,:,:].expand(batch,-1,-1)
        query=self.query_embedding(torch.cat([future,clock],-1))
        attended,weights=self.attention(query.reshape(batch*48,1,-1),tokens,tokens,
            key_padding_mask=mask,need_weights=True,average_attn_weights=True)
        attended=attended.reshape(batch,48,-1)
        weights=weights.reshape(batch,48,self.days,2*self.radius+1)
        context=self.context_norm(query+attended)
        statistics=torch.cat([daily.mean(1),daily[:,-1],daily.std(1,unbiased=False)],-1)
        history=self.history_head(torch.cat([context,query,statistics,future,clock],-1)).squeeze(-1)
        return history,weights

    def forward(self,past,future):
        return self.forward_with_attention(past,future)[0]


def make_daan(spec, past_dim, future_dim, history_steps=336):
    # In archived checkpoints, the original identifier denoted several architectures.
    if spec["model"] == "lead_fusion" and not (
        spec.get("variant") in ("daan", "slot_attention_tcn")
        and spec.get("ablation") == "no_tcn"
        and spec.get("daan_control", "full") == "full"
    ):
        raise ValueError("Expected the paper's DAAN architecture in the archived checkpoint")
    if (history_steps, spec.get("slot_width", 24), spec.get("attention_heads", 4),
            spec.get("slot_radius", 1)) != (336, 24, 4, 1):
        raise ValueError("DAAN uses seven days, width 24, four heads and a one-slot radius")
    return DAAN(spec, past_dim, future_dim, history_steps)
