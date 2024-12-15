import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from model.model import Generator, Discriminator
from preprocessing.dataset import MyImageFolder
from utils import config
from utils.utils import load_checkpoint, save_checkpoint, plot_examples, show_example, val
from utils.evaluation_metrics import EvaluationMetrics
from utils.loss import VGGLoss, TVLoss

torch.backends.cudnn.benchmark = True


def train(loader, epoch, gen, disc, opt_gen, opt_disc, scheduler_gen_lr, scheduler_disc_lr, mse, bce, vgg_loss,
          warm_up=False):
    loop = tqdm(loader, leave=True)
    n_samples = len(loop)
    epoch_gen_loss = 0.0
    epoch_disc_loss = 0.0
    epoch_warmup_loss = 0.0

    for idx, (lr, gt) in enumerate(loop):
        gt = gt.to(config.DEVICE)
        lr = lr.to(config.DEVICE)

        if warm_up:
            # Warm-up phase: Train generator with MSE loss only
            fake = gen(lr)
            gen_loss = mse(fake, gt)  # Use MSE loss during warm-up

            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            epoch_warmup_loss += gen_loss.item()
        else:
            # Train Discriminator: maximize D(x)-1-D(G(z))
            fake = gen(lr)
            disc_real = disc(gt)
            disc_fake = disc(fake.detach())
            disc_loss_real = bce(
                disc_real, torch.ones_like(disc_real)
            )
            disc_loss_fake = bce(
                disc_fake, torch.zeros_like(disc_fake)
            )

            disc_loss = disc_loss_real + disc_loss_fake

            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()

            # Train Generator: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            disc_fake = disc(fake)
            adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))  # Adversial Loss
            loss_for_vgg = 0.006 * vgg_loss(fake, gt)  # Content loss
            mse_loss = mse(fake, gt)  # Image Loss (MSE)
            tv_loss = TVLoss()
            loss_for_tv = 2e-8 * tv_loss(fake)  # Total Variance Loss

            gen_loss = mse_loss + loss_for_vgg + adversarial_loss + loss_for_tv

            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            epoch_gen_loss += gen_loss.item()
            epoch_disc_loss += disc_loss.item()

    return epoch_warmup_loss, epoch_gen_loss, epoch_disc_loss


def main():
    dataset = MyImageFolder(root_gt=config.ROOT_GT, root_lr=config.ROOT_LR, transform=True)
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS
    )
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    scheduler_gen_lr = MultiStepLR(opt_gen, milestones=[3 * (config.NUM_EPOCHS + config.WARMUP_EPOCHS) // 4], gamma=0.1)
    scheduler_disc_lr = MultiStepLR(opt_disc, milestones=[3 * (config.NUM_EPOCHS + config.WARMUP_EPOCHS) // 4],
                                    gamma=0.1)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()
    evaluator = EvaluationMetrics(config.DEVICE)

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_DISC,
            disc,
            opt_disc,
            config.LEARNING_RATE
        )

    max_gen_loss, min_disc_loss = float('inf'), 0

    for epoch in range(1, config.NUM_EPOCHS + config.WARMUP_EPOCHS + 1):
        if epoch <= config.WARMUP_EPOCHS:
            _, _, _ = train(loader, epoch, gen, disc, opt_gen, opt_disc,
                            scheduler_gen_lr, scheduler_disc_lr, mse, bce, vgg_loss, warm_up=True)

        else:
            _, gen_loss, disc_loss = train(loader, epoch, gen, disc, opt_gen, opt_disc,
                                           scheduler_gen_lr, scheduler_disc_lr, mse, bce, vgg_loss, warm_up=False)

        plot_examples(config.ROOT_VAL_LR, config.OUTPUT_VAL_HR, gen)
        dataset_val = MyImageFolder(root_gt=config.ROOT_VAL_GT, root_lr=config.OUTPUT_VAL_HR, transform=False)
        print(val(dataset_val, evaluator, epoch))
        sr_name = os.listdir(config.OUTPUT_VAL_HR)[0]
        show_example(
            lr_path=os.path.join(config.ROOT_VAL_LR, sr_name),
            sr_path=os.path.join(config.OUTPUT_VAL_HR, sr_name),
            gt_path=os.path.join(config.ROOT_VAL_GT, sr_name.replace(f'x{config.SCALE}.png', '.png')),
            epoch=epoch
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, epoch, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, epoch, filename=config.CHECKPOINT_DISC)


if __name__ == "main":
    main()
