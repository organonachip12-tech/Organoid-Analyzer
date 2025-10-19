import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        weights = self.dropout(weights)
        context = torch.sum(weights.unsqueeze(-1) * lstm_out, dim=1)
        return context, weights

class UnifiedFusionModel(nn.Module):
    def __init__(self, seq_input_size, track_input_size, hidden_size=128, fusion_size=128, dropout=0.5):
        super().__init__()
        self.track_input_size = track_input_size

        self.lstm = nn.LSTM(input_size=seq_input_size, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_size*2, dropout)
        self.norm = nn.LayerNorm(hidden_size*2)

        track_output_size = hidden_size*2

        print("Total Sequence parameters:", count_params(self.lstm) + count_params(self.attn) + count_params(self.norm))

        if track_input_size > 0:
            self.track_fc = nn.Sequential(
                nn.Linear(track_input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, track_output_size),
            )
            print("Total Track parameters:", count_params(self.track_fc))

            self.use_track = True
        else:
            self.use_track = False

        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_size * 2 + track_output_size, fusion_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_size, 3)
        )

    def forward(self, x_seq, x_track, lstm_weight=0.5):
        lstm_out, _ = self.lstm(x_seq)
        lstm_out = self.norm(lstm_out)
        lstm_feat, attn_weights = self.attn(lstm_out)
        lstm_feat = F.layer_norm(lstm_feat, lstm_feat.shape[1:])
        lstm_feat = lstm_feat * lstm_weight
        

        if self.use_track:
            track_feat = self.track_fc(x_track)
            track_feat = F.layer_norm(track_feat, track_feat.shape[1:])
            track_feat = track_feat * (1 - lstm_weight)
            fused = torch.cat([lstm_feat, track_feat], dim=1)
        else:
            fused = lstm_feat
        return self.fusion_fc(fused)
    
def count_params(module):
    return sum(p.numel() for p in module.parameters())