

from diffusers import StableDiffusionXLPipeline

import torch
import subprocess
import argparse
import os

from version import BASE_WEIGHTS_URI
from version import BASE_WEIGHTS_PATH

version_weights_uri = BASE_WEIGHTS_URI
version_model_path = BASE_WEIGHTS_PATH

def run(args):
    pipe = StableDiffusionXLPipeline.from_single_file(version_model_path)
    pipe = pipe.to("cuda")

    image_name = args.image_name

    words = image_name.split('.')[0].split('_')
    prompt = ", ".join(words[:-2])
    seeds = int(words[-1])

    generator = torch.Generator("cuda").manual_seed(seeds)
    #generator = torch.Generator().manual_seed(seeds)

    image = pipe(prompt, generator=generator,
                num_inference_steps=40).images[0]
    image.save(f"reproduce.jpg")

def main():
    parser = argparse.ArgumentParser(description="Reproduce")
    parser.add_argument('image_name', type=str, help="The name of the image file to be reproduced.")
    args = parser.parse_args()

    if os.path.exists(version_model_path):
        print(f"version_model_weights already exists.")
    else:
        print(f"version_model_weights does not exist. Downloading from version_weights_uri.")
        subprocess.call("wget -O {version_model_path} {version_weights_uri}", shell=True)

    run(args)

if __name__ == "__main__":
    main()
