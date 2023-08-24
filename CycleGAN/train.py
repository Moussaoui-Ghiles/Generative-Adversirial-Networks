

import torch
from dataset import HorseAndZebraDataset
import sys

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from Discriminator import Discriminator
from Generator import Generator


def train_fn(
    disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse
):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave = True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators H and Z

        fake_horse = gen_H(zebra)
        D_H_real = disc_H(horse)
        D_H_fake = disc_H(fake_horse.detach())
        H_reals += D_H_real.mean().item()
        H_fakes += D_H_fake.mean().item()
        D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
        D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
        D_H_loss = D_H_real_loss + D_H_fake_loss

        fake_zebra = gen_Z(horse)
        D_Z_real = disc_Z(zebra)
        D_Z_fake = disc_Z(fake_zebra.detach())
        D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
        D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
        D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
        D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()


        # Train Generators H and Z

            # adversarial loss for both generators
        D_H_fake = disc_H(fake_horse)
        D_Z_fake = disc_Z(fake_zebra)
        loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
        loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
        cycle_zebra = gen_Z(fake_horse)
        cycle_horse = gen_H(fake_zebra)
        cycle_zebra_loss = l1(zebra, cycle_zebra)
        cycle_horse_loss = l1(horse, cycle_horse)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
        identity_zebra = gen_Z(zebra)
        identity_horse = gen_H(horse)
        identity_zebra_loss = l1(zebra, identity_zebra)
        identity_horse_loss = l1(horse, identity_horse)

            # add all togethor
        G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + cycle_horse_loss * config.LAMBDA_CYCLE
                + identity_horse_loss * config.LAMBDA_IDENTITY
                + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()


        if idx % 200 == 0:
            save_image(fake_horse * 0.5 + 0.5, f"/results/fake_horse_{idx}.png")
            save_image(fake_zebra * 0.5 + 0.5, f"/results/fake_zebra_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()



    dataset = HorseAndZebraDataset(
        root_horse=config.TRAIN_DIR + "/trainA",
        root_zebra=config.TRAIN_DIR + "/trainB",
        transform=config.transforms,
    )
    val_dataset = HorseAndZebraDataset(
        root_horse=config.TRAIN_DIR + "/testA",
        root_zebra=config.TRAIN_DIR + "/testB",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )



    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_H,
            disc_Z,
            gen_Z,
            gen_H,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse

        )




main()