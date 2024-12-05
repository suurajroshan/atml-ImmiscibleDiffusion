import torch
from diffusers import StableDiffusionPipeline
from torchvision.transforms import v2

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = ["a photo of an astronaut" , "photo of a horse on mars"]
out = pipe(prompt, num_inference_steps=20).images

for i, img in enumerate(out):
    img.save(f'{i}.png')

print(torch.stack(v2.functional.pil_to_tensor(out)))


