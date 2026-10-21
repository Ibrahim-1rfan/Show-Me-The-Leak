"""Choose native CUDA arithmetic instead of accepting BF16 emulation on T4."""
def native_dtype(torch):
    return torch.bfloat16 if torch.cuda.get_device_capability()[0]>=8 else torch.float16
