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


# decoder accuracy on test set
def get_decoder_accuracy(full_vid_embedder, full_neu_embedder, test_loader, modality, device=torch.device("cpu")):
    real_fr = dict()
    pred_fr = dict()

    to_pil = T.ToPILImage()
    for i, (image, targets) in (enumerate(test_loader)):
        n_neurons = targets.shape[1]
        image_path = imagehash.average_hash(to_pil(image[:, :, 0].squeeze()))
        image = image.to(device)
        real_fr.setdefault(image_path,[]).append(targets.detach().numpy())
        if not image_path in pred_fr.keys():
            pred_fr.setdefault(image_path,[]).append(full_vid_embedder(image).cpu().detach().numpy())

    n_uq = len(real_fr.keys())

    neu_embeds = np.zeros((n_uq, full_neu_embedder.embed_size))
    vid_embeds = np.zeros((n_uq, full_neu_embedder.embed_size))

    for i, key in enumerate(pred_fr.keys()):
        neu_embeds[i, :] = (full_neu_embedder(torch.tensor(np.mean(np.array(real_fr[key]), 0).squeeze()).to(device))).detach().cpu().numpy()
        vid_embeds[i, :] = (np.array(pred_fr[key]).squeeze())
    
    return top_k_accuracy(neu_embeds, vid_embeds, k=10)

# top k accuracy --- written by ChatGPT
def top_k_accuracy(neu_embeds, vid_embeds, k):
    """
    Compute the top-K accuracy of matching neu_embeds to vid_embeds based on L2 distance.

    Args:
        neu_embeds (np.ndarray): Array of shape (num_examples, embed_size) for neural embeddings.
        vid_embeds (np.ndarray): Array of shape (num_examples, embed_size) for video embeddings.
        k (int): The value of K for top-K accuracy.

    Returns:
        float: Top-K accuracy as a fraction.
    """
    # Compute pairwise L2 distances
    pairwise_dist = np.linalg.norm(neu_embeds[:, None, :] - vid_embeds[None, :, :], axis=2)  # Shape: (num_examples, num_examples)
    
    # Get the indices of the top K closest vid_embed for each neu_embed
    top_k_indices = np.argsort(pairwise_dist, axis=1)[:, :k]  # Shape: (num_examples, K)
    
    # Check if the correct match is within the top K
    correct_matches = np.arange(len(neu_embeds))[:, None]  # Shape: (num_examples, 1)
    hits = np.any(top_k_indices == correct_matches, axis=1)  # Shape: (num_examples,)
    
    # Compute the top-K accuracy
    top_k_accuracy = np.mean(hits)
    return top_k_accuracy
