import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

class SparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size=8):
        super(SparseAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        # 假设块大小为block_size，意味着注意力矩阵按块进行稀疏处理
        seq_len = query.size(0)
        
        # 生成一个稀疏的mask，局部注意力
        attn_mask = self.generate_sparse_mask(seq_len)

        # 执行带有mask的注意力
        attn_output, attn_output_weights = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        
        return attn_output

    def generate_sparse_mask(self, seq_len):
        # 生成一个稀疏的mask，假设每个token只与它前后的block_size个token有联系
        mask = torch.ones(seq_len, seq_len).bool()

        for i in range(seq_len):
            start_idx = max(0, i - self.block_size)
            end_idx = min(seq_len, i + self.block_size)
            mask[i, start_idx:end_idx] = 0
        
        return mask

# # 示例使用
# embed_dim = 64
# num_heads = 4
# seq_len = 16

# query = torch.randn(seq_len, 1, embed_dim)
# key = torch.randn(seq_len, 1, embed_dim)
# value = torch.randn(seq_len, 1, embed_dim)

# sparse_attn = SparseAttention(embed_dim, num_heads)
# # output = sparse_attn(query, key, value)

# # print(output.shape)  # 输出应该是 (seq_len, 1, embed_dim)
# torchinfo.summary(sparse_attn, input_data=(query, key, value))  

import torch
from linformer_pytorch import Linformer
 
# 定义模型参数
dim_head = 64
heads = 8
depth = 12
max_seq_len = 512
 
# 创建 Linformer 模型
model = Linformer(
    seq_len=max_seq_len,
    depth=depth,
    heads=heads,
    k=256
)
 
# 生成随机输入数据
input_data = torch.randn(1, max_seq_len, dim_head * heads)
 
# 前向传播
output = model(input_data)
print(output.shape)  # 输出: torch.Size([1, 512, 512])
