import torch
print("您的torch版本：",torch.__version__)
print("gpu是否可用，True代表可以,False代表不可以：",torch.cuda.is_available())