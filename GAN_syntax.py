import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class TrainAudioSpectrogramDataset(Dataset):
    def __init__(self, root_dir, categories, max_frames=512, fraction=1.0):
        self.root_dir = root_dir
        self.categories = categories
        self.max_frames = max_frames
        self.file_list = []
        self.class_to_idx = {cat: i for i, cat in enumerate(categories)}

        for cat_name in self.categories:
            cat_dir = os.path.join(root_dir, cat_name)
            files_in_cat = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith(".wav")]
            num_to_sample = int(len(files_in_cat) * fraction)
            sampled_files = random.sample(files_in_cat, num_to_sample)
            label_idx = self.class_to_idx[cat_name]
            self.file_list.extend([(file_path, label_idx) for file_path in sampled_files])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=256, n_mels=128
        )(wav)
        log_spec = torch.log1p(mel_spec)

        _, _, n_frames = log_spec.shape
        if n_frames < self.max_frames:
            pad = self.max_frames - n_frames
            log_spec = F.pad(log_spec, (0, pad))
        else:
            log_spec = log_spec[:, :, :self.max_frames]

        label_vec = F.one_hot(torch.tensor(label), num_classes=len(self.categories)).float()
        return log_spec, label_vec

class CGAN_Generator(nn.Module):
    def __init__(self, latent_dim, num_categories, spec_shape=(128, 512)):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.spec_shape = spec_shape

        self.fc = nn.Linear(latent_dim + num_categories, 256 * 8 * 32)
        self.unflatten_shape = (256, 8, 32)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.ReLU()
        )

    def forward(self, z, y):
        h = torch.cat([z, y], dim=1)
        h = self.fc(h)
        h = h.view(-1, *self.unflatten_shape)
        fake_spec = self.net(h)
        return fake_spec

class CGAN_Discriminator(nn.Module):
    def __init__(self, num_categories, spec_shape=(128, 512)):
        super().__init__()
        self.num_categories = num_categories
        self.spec_shape = spec_shape
        H, W = spec_shape

        self.label_embedding = nn.Linear(num_categories, H * W)

        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, (8, 32))
        )

    def forward(self, spec, y):
        label_map = self.label_embedding(y).view(-1, 1, *self.spec_shape)
        h = torch.cat([spec, label_map], dim=1)
        logit = self.net(h)
        return logit.view(-1, 1)

def generate_audio_gan(generator, category_idx, num_samples, device, sample_rate=22050):
    generator.eval()
    num_categories = generator.num_categories
    latent_dim = generator.latent_dim

    y = F.one_hot(torch.tensor([category_idx]), num_classes=num_categories).float().to(device)
    z = torch.randn(num_samples, latent_dim, device=device)

    with torch.no_grad():
        log_spec_gen = generator(z, y)

    spec_gen = torch.expm1(log_spec_gen)
    spec_gen = spec_gen.squeeze(1)

    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=1024 // 2 + 1, n_mels=128, sample_rate=sample_rate
    ).to(device)
    linear_spec = inverse_mel(spec_gen)

    griffin = torchaudio.transforms.GriffinLim(
        n_fft=1024, hop_length=256, win_length=1024, n_iter=32
    ).to(device)

    waveform = griffin(linear_spec)
    return waveform.cpu()

def save_audio_only(wav, sample_rate, filename):
    if wav.dim() > 2:
        wav = wav.squeeze(0)
    torchaudio.save(filename, wav, sample_rate=sample_rate)
    print(f"Saved: {filename}")

def compute_gradient_penalty(discriminator, real_samples, fake_samples, labels, device, lambda_gp=10.0):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates, labels)
    grad_outputs = torch.ones_like(d_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    return penalty

def train_gan(generator, discriminator, dataloader, device, categories, epochs, lr, latent_dim):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9))

    os.makedirs("gan_generated_audio", exist_ok=True)

    n_critic = 5
    lambda_gp = 10.0

    for epoch in range(1, epochs + 1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=True)

        for real_specs, labels in loop:
            real_specs = real_specs.to(device)
            labels = labels.to(device)
            batch_size = real_specs.size(0)

            for _ in range(n_critic):
                optimizer_D.zero_grad()
                z = torch.randn(batch_size, latent_dim, device=device)
                fake_specs = generator(z, labels)

                real_validity = discriminator(real_specs, labels)
                fake_validity = discriminator(fake_specs.detach(), labels)

                loss_D_adv = fake_validity.mean() - real_validity.mean()
                gp = compute_gradient_penalty(discriminator, real_specs.data, fake_specs.data, labels, device, lambda_gp)
                loss_D = loss_D_adv + gp

                loss_D.backward()
                optimizer_D.step()

            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_specs_G = generator(z, labels)
            loss_G = -discriminator(fake_specs_G, labels).mean()
            loss_G.backward()
            optimizer_G.step()

            loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

        generator.eval()
        for cat_idx, cat_name in enumerate(categories):
            wav = generate_audio_gan(generator, cat_idx, 1, device)
            fname = f"gan_generated_audio/{cat_name}_ep{epoch}.wav"
            save_audio_only(wav, sample_rate=22050, filename=fname)
        generator.train()

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LATENT_DIM = 100
    EPOCHS = 200
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4

    TRAIN_PATH = "train"
    train_categories = sorted([d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))])
    NUM_CATEGORIES = len(train_categories)

    print(f"Using device: {DEVICE}")
    print(f"Found {NUM_CATEGORIES} categories: {train_categories}")

    train_dataset = TrainAudioSpectrogramDataset(root_dir=TRAIN_PATH, categories=train_categories)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    generator = CGAN_Generator(LATENT_DIM, NUM_CATEGORIES).to(DEVICE)
    discriminator = CGAN_Discriminator(NUM_CATEGORIES).to(DEVICE)

    train_gan(generator, discriminator, train_loader, DEVICE, train_categories, EPOCHS, LEARNING_RATE, LATENT_DIM)
