import torch

real_len = torch.tensor([3, 4, 5, 6, 7, 8, 2, 3]).to('cuda:1')
batch_size = len(real_len)
label_len = 2
max_len = torch.max(real_len)
seq_range_expand = torch.arange(max_len, device=real_len.device).unsqueeze(0).expand(batch_size, max_len)
seq_length_expand = real_len.unsqueeze(1).expand_as(seq_range_expand)
label_length_expand = seq_length_expand - label_len
out_mask = torch.logical_and(
    torch.less(seq_range_expand, seq_length_expand),
    torch.greater_equal(seq_range_expand, label_length_expand)
)
print(out_mask)
x = torch.randn(size=(batch_size, max_len, 2))
print(x)
print(x[out_mask].view(batch_size, label_len, -1))
print(x.shape[0])