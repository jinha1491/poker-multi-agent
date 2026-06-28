import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Adds position information to token embeddings.
    
    Why we need this:
    Transformers process all tokens simultaneously (not sequentially like RNNs).
    Without positional encoding, the model can't tell if 'raise' came first or third
    in the sequence. We inject position info using sine/cosine waves of different
    frequencies — a technique from the original 'Attention is All You Need' paper.
    """
    def __init__(self, d_model: int, max_len: int = 20, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # position indices: 0, 1, 2, ..., max_len-1
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # division term creates different frequencies for each dimension
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # even dimensions get sine, odd dimensions get cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # register as buffer — saved with model but not a trainable parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add positional encoding to token embeddings
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PokerTransformer(nn.Module):
    """
    Transformer encoder that learns opponent behavioral patterns
    from sequences of poker actions.
    
    Architecture:
    Token IDs → Embedding → Positional Encoding → 
    Transformer Encoder Layers → Mean Pooling → 
    Classification Head → Action Probabilities
    """
    def __init__(
        self,
        vocab_size: int = 8,       # 6 actions + PAD + UNK
        d_model: int = 64,         # embedding dimension
        nhead: int = 4,            # number of attention heads
        num_layers: int = 2,       # number of transformer layers
        num_classes: int = 5,      # fold, call, check, raise, bet
        max_len: int = 20,         # max sequence length
        dropout: float = 0.1
    ):
        super().__init__()

        # embedding layer: converts token IDs to dense vectors
        # vocab_size=8 tokens, each mapped to d_model=64 dimensional vector
        # padding_idx=6 means PAD tokens always map to zero vector
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=6)

        # positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # transformer encoder
        # each layer has: multi-head self-attention + feed-forward network
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,   # hidden size of feed-forward network
            dropout=dropout,
            batch_first=True       # input shape: (batch, seq, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # classification head: maps from d_model to num_classes
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: token IDs of shape (batch_size, seq_len)
        padding_mask: True where tokens are PAD, shape (batch_size, seq_len)
        """
        # convert token IDs to embeddings: (batch, seq, d_model)
        x = self.embedding(x)

        # add positional encoding
        x = self.pos_encoding(x)

        # pass through transformer encoder layers
        # padding_mask tells attention to ignore PAD tokens
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # mean pooling: average across sequence dimension
        # collapses (batch, seq, d_model) → (batch, d_model)
        # we ignore PAD positions in the mean
        if padding_mask is not None:
            mask = (~padding_mask).float().unsqueeze(-1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            x = x.mean(dim=1)

        # classify: (batch, d_model) → (batch, num_classes)
        return self.classifier(x)

    def encode(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Returns the embedding before classification.
        Used by the RL agent to get opponent behavioral encoding.
        """
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        if padding_mask is not None:
            mask = (~padding_mask).float().unsqueeze(-1)
            return (x * mask).sum(dim=1) / mask.sum(dim=1)
        return x.mean(dim=1)


if __name__ == "__main__":
    # quick sanity check
    model = PokerTransformer()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # fake batch of 4 sequences, each length 20
    dummy_input = torch.randint(0, 6, (4, 20))
    padding_mask = (dummy_input == 6)  # True where PAD

    output = model(dummy_input, padding_mask)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # should be (4, 5)
    print("Transformer working correctly")
