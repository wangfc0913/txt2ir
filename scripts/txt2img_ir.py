import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.T2IRDataset import T2IRValidation, T2IRTrain


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(img, image_path):
    img = tensor2im(img)
    image_pil = Image.fromarray(img)
    image_pil.save(image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--prompt",
    #     type=str,
    #     nargs="?",
    #     default="a painting of a virus monster playing guitar",
    #     help="the prompt to render"
    # )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2ir-ddpm-mask-llvip-train"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()

    config = OmegaConf.load("configs/latent-diffusion/txt2ir-kl-32x32x4.yaml")
    model = load_model_from_config(config, "models/ldm/model.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        print(f'sampler: {PLMSSampler}')
        sampler = PLMSSampler(model)
    else:
        print(f'sampler: {DDIMSampler}')
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # prompt = opt.prompt


    #sample_path = os.path.join(outpath, "samples")
    #os.makedirs(sample_path, exist_ok=True)
    #base_count = len(os.listdir(sample_path))

    dataset = T2IRTrain(data_name="LLVIP")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1)
    print(f"Found {len(dataloader)} inputs.")

    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():
            for data in tqdm(dataloader):
                out_file_name = os.path.join(outpath, data['file_name'][0] + '_sample.png')
                caption = data["caption"]
                c = model.get_learned_conditioning(caption)
                shape = [4, opt.H//8, opt.W//8]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)

                predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                              min=0.0, max=1.0)
                predicted_image = predicted_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255

                predicted_image = np.squeeze(predicted_image)
                Image.fromarray(predicted_image.astype(np.uint8)).save(out_file_name)

