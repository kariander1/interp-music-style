import torch
from torch import nn

from ldm.data.personalized import per_img_token_list
from transformers import CLIPTokenizer
from functools import partial
import numpy as np
from ldm.modules.attention import CrossAttention
from ldm.modules.diffusionmodules.openaimodel import ResBlock
import PIL
from PIL import Image

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens[0, 1]

def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token

def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]


class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_words=None,
            per_image_tokens=False,
            num_vectors_per_token=1,
            progressive_words=False,
            alpha=None,
            **kwargs
    ):
        super().__init__()
        self.loaded = False
        self.string_to_token_dict = {}
        
        self.string_to_param_dict = nn.ParameterDict()

        self.initial_embeddings = nn.ParameterDict() # These should not be optimized

        self.progressive_words = progressive_words
        self.progressive_counter = 0

        self.max_vectors_per_token = num_vectors_per_token

        if hasattr(embedder, 'tokenizer'): # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings)
            token_dim = 768
        else: # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            get_embedding_for_tkn = embedder.transformer.token_emb
            token_dim = 1280

        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)

        time_embed_dim = token_dim * 4
        self.time_embed = nn.Sequential(
            linear(token_dim, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                time_embed_dim, token_dim),
        )
 
        self.attention = Attentions(dim=token_dim, n_heads=8, d_head=64, dropout = 0.05) 
        self.alpha = alpha
        for idx, placeholder_string in enumerate(placeholder_strings):
            
            token = get_token_for_string(placeholder_string)

            if initializer_words and idx < len(initializer_words):
                init_word_token = get_token_for_string(initializer_words[idx])

                with torch.no_grad():
                    init_word_embedding = get_embedding_for_tkn(init_word_token.cpu())

                token_params = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=True)
                self.initial_embeddings[placeholder_string] = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=False)
            else:
                token_params = torch.nn.Parameter(torch.rand(size=(num_vectors_per_token, token_dim), requires_grad=True))
            
            # self.initial_embedding = torch.rand(size=(num_vectors_per_token, token_dim))
            self.string_to_token_dict[placeholder_string] = token
            self.string_to_param_dict[placeholder_string] = token_params


    def process_embeddings(
        self,
        tokenized_text,
        embedded_text,
        timestep,
        string_to_token_dict,
        initial_embeddings,
        string_to_param_dict,
        max_vectors_per_token,
        progressive_words=False,
        progressive_counter=0,
        emb_layers=None,
        attention=None,
        time_embed=None,
        progressive_scale=1
    ):
        token_dim = embedded_text.shape[2]
        t_emb = timestep_embedding(timestep, token_dim, repeat_only=False)
        emb = time_embed(t_emb)

        b, n, device = *tokenized_text.shape, tokenized_text.device

        for placeholder_string, placeholder_token in string_to_token_dict.items():
            if initial_embeddings[placeholder_string] is None:
                print('Working with NO IMAGE mode')
                # Create a zero embedding if no guidance
                initial_embeddings[placeholder_string] = self.get_embedding_for_tkn('').unsqueeze(0).repeat(
                    max_vectors_per_token, 1).to(device)

            if string_to_param_dict is not None:
                string_to_param_dict[placeholder_string] = torch.nn.Parameter(
                    initial_embeddings[placeholder_string], requires_grad=False).to(device)

            h = emb_layers(emb).view(b, 1, token_dim) + initial_embeddings[placeholder_string]
            placeholder_embedding = attention(h, h).view(b, token_dim)

            if max_vectors_per_token == 1:
                placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
                embedded_text[placeholder_idx] = placeholder_embedding.float()
            else:
                if progressive_words:
                    progressive_counter += 1
                    max_step_tokens = 1 + progressive_counter // progressive_scale
                else:
                    max_step_tokens = max_vectors_per_token

                num_vectors_for_token = min(placeholder_embedding.shape[0], max_step_tokens)

                placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))
                if placeholder_rows.nelement() == 0:
                    continue

                sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                sorted_rows = placeholder_rows[sort_idx]

                for idx in range(len(sorted_rows)):
                    row = sorted_rows[idx]
                    col = sorted_cols[idx]

                    new_token_row = torch.cat([
                        tokenized_text[row][:col],
                        placeholder_token.repeat(num_vectors_for_token).to(device),
                        tokenized_text[row][col + 1:]
                    ], axis=0)[:n]

                    new_embed_row = torch.cat([
                        embedded_text[row][:col],
                        placeholder_embedding[:num_vectors_for_token],
                        embedded_text[row][col + 1:]
                    ], axis=0)[:n]

                    embedded_text[row] = new_embed_row
                    tokenized_text[row] = new_token_row

        return embedded_text, tokenized_text, progressive_counter

    def forward(
            self,
            tokenized_text,
            embedded_text,
            timestep=None
    ):
        if self.loaded:
            embeddings = []
            for i in range(self.n_embeddings):
                self.string_to_token_dict = self.string_to_token_dict_list[i]
                self.attention = self.attention_list[i]
                self.time_embed = self.time_embed_list[i]
                self.emb_layers = self.emb_layers_list[i]
                self.initial_embeddings = self.initial_embeddings_list[i]
                
                embedded_text_temp, _, _ = self.process_embeddings(
                    tokenized_text=tokenized_text,
                    embedded_text=embedded_text,
                    timestep=timestep,
                    string_to_token_dict=self.string_to_token_dict,
                    initial_embeddings=self.initial_embeddings,
                    string_to_param_dict=self.string_to_param_dict,
                    max_vectors_per_token=self.max_vectors_per_token,
                    progressive_words=self.progressive_words,
                    progressive_counter=self.progressive_counter,
                    emb_layers=self.emb_layers,
                    attention=self.attention,
                    time_embed=self.time_embed,
                    progressive_scale=PROGRESSIVE_SCALE
                )
                embeddings.append(embedded_text_temp.clone())
                
            alpha = self.alpha
            if not alpha:
                alpha = [1.0/self.n_embeddings]*self.n_embeddings
            embedded_text = self.interpolate_embeddings(embeddings, alpha=alpha)
        else:
            embedded_text, _, _ = self.process_embeddings(
            tokenized_text=tokenized_text,
            embedded_text=embedded_text,
            timestep=timestep,
            string_to_token_dict=self.string_to_token_dict,
            initial_embeddings=self.initial_embeddings,
            string_to_param_dict=self.string_to_param_dict,
            max_vectors_per_token=self.max_vectors_per_token,
            progressive_words=self.progressive_words,
            progressive_counter=self.progressive_counter,
            emb_layers=self.emb_layers,
            attention=self.attention,
            time_embed=self.time_embed,
            progressive_scale=PROGRESSIVE_SCALE
            )
        return embedded_text
    
    def interpolate_embeddings(self, embeddings, alpha):
        """
        Interpolates a list of embeddings using the provided alpha weights.

        Args:
            embeddings (List[Tensor]): List of tensors to interpolate, each of shape (...).
            alpha (List[float] or Tensor): Interpolation weights, one per embedding.

        Returns:
            Tensor: Interpolated embedding tensor.
        """
        embeddings = torch.stack(embeddings, dim=0)  # Shape: (n, ...)
        alpha = torch.tensor(alpha, dtype=embeddings.dtype, device=embeddings.device)  # Shape: (n,)
        alpha = alpha / alpha.sum()  # Normalize to sum to 1 (optional but often desirable)

        # Add dimensions to alpha for broadcasting
        while alpha.dim() < embeddings.dim():
            alpha = alpha.unsqueeze(-1)

        interpolated = (embeddings * alpha).sum(dim=0)
        return interpolated
    def save(self, ckpt_path):
        torch.save({
                "string_to_token": self.string_to_token_dict,
                "attention": self.attention,
                "time_embed": self.time_embed,
                "emb_layers": self.emb_layers,
                "initial_embeddings": self.initial_embeddings
                }, ckpt_path)
        
    def load(self, ckpt_paths, device):
        if isinstance(ckpt_paths, str):
            ckpt_paths = [ckpt_paths]

        self.string_to_token_dict_list = []
        self.attention_list = []
        self.time_embed_list = []
        self.emb_layers_list = []
        self.initial_embeddings_list = []

        for path in ckpt_paths:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            self.string_to_token_dict_list.append(ckpt["string_to_token"])
            self.attention_list.append(ckpt["attention"].to(device))
            self.time_embed_list.append(ckpt["time_embed"].to(device))
            self.emb_layers_list.append(ckpt["emb_layers"].to(device))
            self.initial_embeddings_list.append(ckpt["initial_embeddings"].to(device))

        self.n_embeddings = len(self.string_to_token_dict_list)
        self.loaded = True
    def get_embedding_norms_squared(self):
        all_params = torch.cat(list(self.string_to_param_dict.values()), axis=0) # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(axis=-1)              # num_placeholders

        return param_norm_squared

    def embedding_parameters(self):
        # return self.string_to_param_dict.parameters()
        return [self.attention.parameters(), self.time_embed.parameters(), self.emb_layers.parameters()]

    def list_embedding_parameters(self):
        params = list(self.attention.parameters()) + list(self.time_embed.parameters()) + list(self.emb_layers.parameters())
        return params

    def embedding_to_coarse_loss(self):
        
        loss = 0.
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        return loss
    
class Attentions(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim))

    def forward(self, x, context=None):
        x_1 = self.attn1(x)
        x_2 = self.attn2(x_1, x)
        x_3 = self.net(x_2)
        return x_3
    
class TimeX(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
    ):
    
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            nn.SiLU(),
            linear(channels, self.out_channels,),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels, self.out_channels,),
        )

        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            linear(self.out_channels, self.out_channels,),
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        h = h + emb_out
        h = self.out_layers(h)

