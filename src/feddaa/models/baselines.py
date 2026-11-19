"""Forecasting baselines with the widths and decoder used in the paper."""

import torch
from torch import nn

from feddaa.data.windows import FORECAST_STEPS


def make_baseline(name, past_dim, future_dim, history_steps, hidden_dim, dropout):
    heads = 4 if hidden_dim % 4 == 0 else 1

    class PerLeadDecoder(nn.Module):
        def __init__(self, context_dim: int):
            super().__init__()
            future_hidden = max(hidden_dim // 2, 32)
            lead_hidden = max(hidden_dim // 8, 8)
            self.future_projection = nn.Sequential(
                nn.Linear(future_dim, future_hidden),
                nn.GELU(),
                nn.LayerNorm(future_hidden),
            )
            self.lead_embedding = nn.Embedding(FORECAST_STEPS, lead_hidden)
            self.head = nn.Sequential(
                nn.Linear(context_dim + future_hidden + lead_hidden, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, max(hidden_dim // 2, 32)),
                nn.GELU(),
                nn.Linear(max(hidden_dim // 2, 32), 1),
            )

        def forward(self, context, future):
            context = context.unsqueeze(1).expand(-1, FORECAST_STEPS, -1)
            future_context = self.future_projection(future)
            leads = torch.arange(FORECAST_STEPS, device=future.device)
            lead_context = self.lead_embedding(leads).unsqueeze(0).expand(len(future), -1, -1)
            return self.head(torch.cat([context, future_context, lead_context], dim=-1)).squeeze(-1)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.history = nn.Sequential(
                nn.Flatten(),
                nn.Linear(history_steps * past_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            )
            self.decoder = PerLeadDecoder(hidden_dim)

        def forward(self, past, future):
            return self.decoder(self.history(past), future)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            channels = past_dim
            for dilation in (1, 2, 4, 8, 16, 32, 64):
                layers.extend(
                    [
                        nn.Conv1d(channels, hidden_dim, 3, padding=dilation, dilation=dilation),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
                channels = hidden_dim
            self.encoder = nn.Sequential(*layers)
            self.context = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
            self.decoder = PerLeadDecoder(hidden_dim)

        def forward(self, past, future):
            encoded = self.encoder(past.transpose(1, 2))
            pooled = torch.cat([encoded.mean(dim=-1), encoded[:, :, -1]], dim=-1)
            return self.decoder(self.context(pooled), future)

    class TiDE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(history_steps * past_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.future_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(FORECAST_STEPS * future_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, FORECAST_STEPS),
            )

        def forward(self, past, future):
            return self.decoder(torch.cat([self.encoder(past), self.future_encoder(future)], dim=-1))

    class TimeXer(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_len = 16
            self.stride = 8
            self.patch_count = max(1, (history_steps - self.patch_len) // self.stride + 1)
            self.endogenous_projection = nn.Linear(self.patch_len, hidden_dim)
            self.endogenous_global = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.endogenous_position = nn.Parameter(torch.zeros(1, self.patch_count + 1, hidden_dim))
            self.exogenous_history = nn.Linear(history_steps, hidden_dim)
            self.future_projection = nn.Linear(FORECAST_STEPS, hidden_dim)
            self.future_var_embedding = nn.Parameter(torch.zeros(1, future_dim, hidden_dim))
            self.past_var_embedding = nn.Parameter(torch.zeros(1, max(past_dim - 1, 1), hidden_dim))
            self.self_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    hidden_dim,
                    heads,
                    hidden_dim * 4,
                    dropout,
                    batch_first=True,
                    norm_first=False,
                ),
                num_layers=2,
            )
            self.cross_attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)
            self.context = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
            self.decoder = PerLeadDecoder(hidden_dim)

        def forward(self, past, future):
            endogenous = past[:, :, 0].unfold(dimension=1, size=self.patch_len, step=self.stride)
            endogenous = self.endogenous_projection(endogenous.contiguous())
            global_token = self.endogenous_global.expand(len(past), -1, -1)
            endogenous_tokens = torch.cat([global_token, endogenous], dim=1) + self.endogenous_position
            endogenous_tokens = self.self_encoder(endogenous_tokens)
            query = endogenous_tokens[:, :1]

            if past_dim > 1:
                exog_history = self.exogenous_history(past[:, :, 1:].transpose(1, 2))
                exog_history = exog_history + self.past_var_embedding[:, : past_dim - 1]
            else:
                exog_history = query.new_zeros(len(past), 1, query.shape[-1])
            future_tokens = self.future_projection(future.transpose(1, 2)) + self.future_var_embedding
            exog_tokens = torch.cat([exog_history, future_tokens], dim=1)
            attended, _ = self.cross_attention(query=query, key=exog_tokens, value=exog_tokens)
            global_context = self.norm(query + attended).squeeze(1)
            patch_context = endogenous_tokens[:, 1:].mean(dim=1)
            return self.decoder(self.context(torch.cat([global_context, patch_context], dim=-1)), future)

    models = {"mlp": MLP, "tcn": TCN, "tide": TiDE, "timexer": TimeXer}
    return models[name]()
