import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dataset_load import *

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(64, 64, 3), cmap='Greys_r')
    return fig



def model_train_test(model_G, model_D, optimizer_D, optimizer_G, epochs, z_size, batch_size, criterion, data_loader, real_label, fake_label, device):
    label_real = torch.full((batch_size,), real_label, device=device, dtype=torch.float)
    label_fake = torch.full((batch_size,), fake_label, device=device, dtype=torch.float)

    fixed_noise = torch.randn(batch_size, z_size, 1, 1, device=device, dtype=torch.float)

    for epoch in range(epochs):
        model_G.train()
        model_D.train()

        for i, data in enumerate(data_loader):
            data = data[0].to(device)

            noise = torch.randn(batch_size, z_size, 1, 1, device=device, dtype=torch.float)
            fake_data = model_G(noise)

            # Discriminator 학습
            model_D.zero_grad()

            output_real = model_D(data).view(-1)
            Loss_D_real = criterion(output_real, label_real)
            Loss_D_real.backward()

            output_fake = model_D(fake_data.detach()).view(-1)
            Loss_D_fake = criterion(output_fake, label_fake)
            Loss_D_fake.backward()

            Loss_D = Loss_D_real + Loss_D_fake
            optimizer_D.step()


            # Generator 학습
            model_G.zero_grad()

            output = model_D(fake_data).view(-1)
            Loss_G = criterion(output, label_real)

            Loss_G.backward()
            optimizer_G.step()


            # Output training stats
            if i % 400 == 0 and i != 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                    % (epoch, epochs, i, len(data_loader),
                        Loss_D.item(), Loss_G.item()))

                model_G.eval()
                model_D.eval()
                with torch.no_grad():
                    output = model_G(fixed_noise).detach().cpu().numpy()
                    output = np.transpose((output+1)/2, (0, 2, 3, 1))
                    fig = plot(output[:16])

                model_G.train()
                model_D.train()


