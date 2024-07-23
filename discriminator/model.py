import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_labels=3):
        super(Discriminator, self).__init__()
        self.Wn = nn.Linear(embed_dim, embed_dim)
        self.Wr = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.attention = MultiHeadAttention(embed_dim, num_heads=num_heads)

        self.fc = nn.Linear(embed_dim, num_labels)

    def forward(self, word, rel, mask):
        # word: [batch_size, word_num, embed_dim]
        # rel: [batch_size, rel_num, embed_dim]
        # mask: [batch_size, rel_num]

        # [batch_size, word_num, rel_num]
        mask = mask.unsqueeze(1).expand(-1, word.size(1), -1)

        # Put word and rel through projection matrix
        word = self.Wn(word)
        rel = self.Wr(rel)

        # [batch_size, word_num, rel_num, embed_dim]
        word_exp = word.unsqueeze(2).expand(-1, -1, rel.size(1), -1)
        # [batch_size, word_num, rel_num, embed_dim]
        rel_exp = rel.unsqueeze(1).expand(-1, word.size(1), -1, -1)

        # [batch_size, word_num, rel_num, embed_dim]
        hadamard = word_exp * rel_exp
        hadamard = self.dropout(self.relu(hadamard))

        # [batch_size, embed_dim]
        hadamard = self.attention(hadamard, mask)
        # hadamard = self.dropout(hadamard)
        hadamard = self.dropout(self.relu(hadamard))

        # [batch_size, num_labels]
        logits = self.fc(hadamard)

        return logits


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(0.2)

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embed_dim must be divisible by num_heads"

        self.linear = nn.ModuleList(
            [nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.W = nn.ModuleList([nn.Linear(self.head_dim, 1) for _ in range(num_heads)])
        self.final_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask):
        # x: [batch_size, word_num, rel_num, embed_dim]
        # mask: [batch_size, word_num, rel_num]
        heads = []
        for i in range(self.num_heads):
            # transformed_x: [batch_size, word_num, rel_num, head_dim]
            transformed_x = self.dropout(self.linear[i](x))
            # weights: [batch_size, word_num, rel_num]
            weights = torch.softmax(self.W[i](transformed_x).squeeze(-1), dim=-1)
            weights = weights * mask

            # head: [batch_size, word_num, head_dim]
            head = (weights.unsqueeze(-1) * transformed_x).sum(dim=2)
            head = head.mean(dim=1)  # [batch_size, head_dim]
            heads.append(head)
        output = torch.cat(heads, dim=-1)  # [batch_size, embed_dim]
        output = self.final_linear(output)  # [batch_size, embed_dim]
        return output
