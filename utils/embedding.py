import torch


class Embedding(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
        """Construct an embedding module. This function should accept the following
           parameters:

           Args:
               num_embeddings: Size of the vocabulary
               embedding_dim: Dimension of the embedding vectors, i.e., dmodel
               device: Device to store the parameters on
               dtype: Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        num_embed_tensor = torch.tensor(num_embeddings, device=device, dtype=torch.float32)
        std = 1.0 / torch.sqrt(num_embed_tensor)
        a_tensor = torch.tensor(-2.0 * std.item(), device=device, dtype=dtype)
        b_tensor = torch.tensor(2.0 * std.item(), device=device, dtype=dtype)
        torch.nn.init.normal_(self.weight, mean=0.0, std=std.item())
        self.weight.data.clamp_(min=a_tensor, max=b_tensor)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors."""
        # Standard Python lists and NumPy arrays) does not inherently support the syntax of using
        # an entire array (or tensor) as an index to extract multi-dimensional sub-elements.
        #
        # This syntax, such as weights[token_ids], is known as Advanced Indexing (or Fancy Indexing),
        # and it is provided by specialized libraries designed for processing large amounts of data.
        #
        # weights[5] → embedding of vocab 5
        # weights[[5, 8, 12]] → return embedding of 5, 8, 12 at the same time
        return self.weight[token_ids]

