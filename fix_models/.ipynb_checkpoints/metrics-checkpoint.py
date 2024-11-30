import torch
from scipy.stats import spearmanr, pearsonr
import numpy as np
import imagehash
import torchvision.transforms as T

# correlation to average on test set
def corr_to_avg(model, test_loader, modality, device=torch.device("cpu")):
    real_fr = dict()
    pred_fr = dict()

    to_pil = T.ToPILImage()
    for i, (image, targets) in (enumerate(test_loader)):
        n_neurons = targets.shape[1]
        if modality == "image":
            image_path = imagehash.average_hash(to_pil(image.squeeze()))
        else:
            image_path = imagehash.average_hash(to_pil(image[:, :, 0].squeeze()))
        image = image.to(device)
        real_fr.setdefault(image_path,[]).append(targets.detach().numpy())
        if not image_path in pred_fr.keys():
            pred_fr.setdefault(image_path,[]).append(model(image).cpu().detach().numpy())

    n_uq = len(real_fr.keys())

    avg = np.zeros(((n_uq, n_neurons)))
    pred = np.zeros(((n_uq, n_neurons)))

    for i, key in enumerate(pred_fr.keys()):
        avg[i, :] = (np.mean(np.array(real_fr[key]).squeeze(), 0))
        pred[i, :] = (np.array(pred_fr[key]).squeeze())
    
    return pearsonr(pred, avg, axis=0).statistic