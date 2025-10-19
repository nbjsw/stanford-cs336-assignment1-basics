import torch

from . import embedding, prenorm_transformer_block, rmsnorm, linear


class TransformerLM(torch.nn.Module):

    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 device=None,
                 dtype=None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.token_embeddings = embedding.Embedding(vocab_size, d_model, **factory_kwargs)

        self.layers = torch.nn.ModuleList(
            [prenorm_transformer_block.PreNormTransformerBlock(
                 d_model, num_heads, d_ff, context_length, rope_theta, **factory_kwargs) for _ in range(num_layers)]
        )

        self.ln_final = rmsnorm.RMSNorm(d_model, **factory_kwargs)

        self.lm_head = linear.Linear(d_model, vocab_size)

    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

