import      torch
import      torch.nn.functional as F
import      torch.nn as nn
import      torch.optim as optim
import      torchvision
import      torchvision.transforms as transforms
from        torchvision.models import resnet18
from        torch.utils.data import DataLoader
import      numpy as np
#from        sklearn.cluster import KMeans
import      matplotlib.pyplot as plt
import      seaborn as sns
#import      os
from        utils               import * 
#from        sklearn.manifold    import TSNE

if u_detect_environment()[0]:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimCLR(nn.Module):
    def __init__(self, base_model=resnet18, feature_dim=128):
        super(SimCLR, self).__init__()
        self.backbone = base_model(pretrained=True)
        self.backbone.fc = nn.Identity() 
        self.projection = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        features      = self.backbone(x)
        projections   = self.projection(features)
        return features, projections



def nt_xent_loss(z_i, z_j, temperature=0.5):

    device      = z_i.device
    batch_size  = z_i.shape[0]

    z       = torch.cat([z_i, z_j], dim=0)  
    z       = F.normalize(z, dim=1)
    sim_matrix = torch.mm(z, z.T)  

    labels  = torch.arange(batch_size, device=device)
    labels  = torch.cat([labels + batch_size, labels])  

    logits  = sim_matrix / temperature

    mask    = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    logits  = logits.masked_fill(mask, -float("inf"))

    loss    = F.cross_entropy(logits, labels)
    return loss


class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


def train_simclr(model, dataloader, optimizer, epochs=10):
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for (x, _) in tqdm(dataloader):
            optimizer.zero_grad()

            x_i, x_j = x[0].to(device), x[1].to(device)

            _, z_i = model(x_i)
            _, z_j = model(x_j)

            loss = nt_xent_loss(z_i, z_j)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
   

def extract_embeddings(model, dataloader):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for (x, y) in tqdm(dataloader):
            x = x.to(device)
            features, _ = model(x)  
            embeddings.append(features.cpu().numpy())
            labels.extend(y.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, np.array(labels)


################################################################################    
################################################################################    
################################################################################    
def Fext_contrastive(confs):

    lconfs      = confs.feat_ext
    db_pt       = confs.db_pt.colab if u_detect_environment()[0] else confs.db_pt.local
    data_pt     = f'{confs.local_pt.colab if u_detect_environment()[0] else confs.local_pt.local}/data'  

    #--------------------------------------------------------------------------- 
    contrast_transforms = transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Resize((lconfs.im_size, lconfs.im_size)),
                                        #   transforms.RandomApply([
                                        #       transforms.ColorJitter(brightness=0.1,
                                        #                              contrast=0.1,
                                        #                              saturation=0.1,
                                        #                              hue=0.1)
                                        #   ], p=0.8),
                                        #   transforms.RandomGrayscale(p=0.05),
                                            transforms.GaussianBlur(kernel_size=5),
                                            transforms.ToTensor()
                                         ])


    dataset     = torchvision.datasets.ImageFolder( root=db_pt,
                                                    transform=ContrastiveTransformations(contrast_transforms, n_views=2))
    
    dataloader  = DataLoader(dataset, batch_size=lconfs.batch_size, shuffle=True, num_workers=lconfs.num_workers)

    model       = SimCLR().to(device)

    optimizer   = optim.AdamW(model.parameters(), lr=3e-4)

    #--------------------------------------------------------------------------- 
    # ite         = iter(dataloader)

    # n_images    = 9
    # fig, ax = plt.subplots(2, n_images, figsize=(12, 5))
    
    # for col, i in enumerate(range (0, n_images)):
    #     x, _ = next(ite)
    #     ax[0][i].imshow(x[0][0].permute(1, 2, 0)) 
    #     ax[1][i].imshow(x[1][0].permute(1, 2, 0))

    # plt.show()
    # plt.close()
    
    train_simclr(model, dataloader, optimizer, epochs=lconfs.epochs)

    #---------------------------------------------------------------------------
    version_pt   = f'{data_pt}/{u_getLastFile(data_pt, lconfs.model, lconfs.new_folder)}'
    u_mkdir(version_pt)
    model_pt     = f'{version_pt}/model.pt'
    history_pt   = f'{version_pt}/history.yml'

    u_saveYaml(history_pt, u_class2dict(lconfs))
    print(f"Saving model to {model_pt}")
    torch.save(model.state_dict(), model_pt)

    
