import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from scipy.stats import kendalltau, spearmanr
from skimage.segmentation import slic
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import json
import torch
import numpy as np
from PIL import Image


def pgd_attack(model, image, labels, original_explanations, eps=8/255, alpha=2/255, iters=10, beta=0.5):
    perturbed_image = image.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    perturbed_image.requires_grad = True

    for i in range(iters):
        # Zero all existing gradients
        model.zero_grad()
        if perturbed_image.grad is not None:
            perturbed_image.grad.data.zero_()

        # Forward pass
        outputs = model(perturbed_image)

        # Prediction loss: negative cross-entropy to keep predictions the same
        loss_pred = -nn.CrossEntropyLoss()(outputs, labels)

        # Explanation attribution
        # Compute explanations for current perturbed_images
        model.zero_grad()
        perturbed_image.grad = None
        loss_expl = outputs.gather(1, labels.unsqueeze(1)).squeeze()
        loss_expl.backward(retain_graph=True)
        attributions = perturbed_image.grad.data.clone()

        # Compute negative MSE between explanations
        mse_loss = -F.mse_loss(attributions, original_explanations)

        # Total loss
        loss = loss_pred + beta * mse_loss

        model.zero_grad()
        perturbed_image.grad.data.zero_()
        loss.backward()

        grad = perturbed_image.grad.data
        perturbed_image = perturbed_image + alpha * grad.sign()

        perturbed_image = torch.max(
            torch.min(perturbed_image, image + eps), image - eps)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        perturbed_image = perturbed_image.detach()
        perturbed_image.requires_grad = True

        print(
            f"Iteration {i+1}: Loss Pred: {loss_pred.item():.4f}, Loss Expl: {mse_loss.item():.4f}")

    return perturbed_image


def lime_explain(input_tensor, model, predicted_idx, num_segments=50, features_pct=0.1):
    model.eval()

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    input_image = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    input_image = np.clip(input_image, 0, 1)

    # Create superpixels using SLIC
    segments = slic(
        input_image,
        n_segments=num_segments,
        compactness=10,
        sigma=1,
        start_label=0
    )

    actual_segments = np.unique(segments).size
    if actual_segments < num_segments:
        print(
            f"Warning: Requested {num_segments} segments, but SLIC returned {actual_segments} segments.")
        num_segments = actual_segments  # Adjust to actual number

    perturbations = []
    perturbed_images = []

    # Generate perturbations by turning superpixels on/off
    for i in range(num_segments):
        # Create a binary mask: 1 for active superpixels, 0 for the perturbed superpixel
        perturb = np.ones(num_segments, dtype=int)
        perturb[i] = 0  # Deactivate superpixel i
        perturbations.append(perturb)

        # Create perturbed image by blacking out superpixel i
        perturbed_image = input_image.copy()
        perturbed_image[segments == i] = 0  # Black out superpixel
        perturbed_images.append(perturbed_image)

    # Convert lists to NumPy arrays
    perturbations = np.array(perturbations)
    perturbed_images = np.array(perturbed_images)

    # Flatten original image for similarity computation
    original_flat = input_image.flatten()

    # Create weights based on similarity to original image
    weights = np.array([
        np.exp(-np.linalg.norm(img.flatten() - original_flat) / 1000)
        for img in perturbed_images
    ])

    # Predict using the model
    with torch.no_grad():
        perturb_tensor = torch.tensor(perturbed_images).permute(
            0, 3, 1, 2).float().to(device)
        predictions = model(perturb_tensor)
        y = predictions[:, predicted_idx].cpu().numpy()

    reg = LinearRegression()
    reg.fit(perturbations, y, sample_weight=weights)

    # Get explanations (coefficients)
    explanations = reg.coef_

    # Get superpixel labels
    superpixel_labels = np.unique(segments)
    superpixel_labels.sort()

    # Map explanations to superpixel labels
    lime_weights = dict(zip(superpixel_labels, explanations))

    # Select top contributing superpixels
    num_top = int(len(explanations) * features_pct)
    top_indices = np.argsort(np.abs(explanations))[-num_top:]

    # Get the labels of top superpixels
    top_superpixels = superpixel_labels[top_indices]

    # Create mask for top superpixels
    mask = np.isin(segments, top_superpixels)

    # Apply gray mask to non-contributing regions
    highlighted_image = input_image.copy()
    gray_color = [0.5, 0.5, 0.5]  # Gray in normalized [0,1] range
    highlighted_image[~mask] = gray_color

    # Convert to PIL Image
    highlighted_pil = Image.fromarray(
        (highlighted_image * 255).astype(np.uint8))

    return lime_weights, segments, highlighted_pil


def smoothed_gradients(input_tensor, model, predicted_idx, segments, num_samples=50, stdev_spread=0.15):

    device = next(model.parameters()).device

    # Ensure input_tensor has batch dimension
    if input_tensor.dim() == 3:
        # Shape: [1, channels, height, width]
        input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    # Convert tensor to numpy array for visualization
    input_image = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    input_image = np.clip(input_image, 0, 1)

    stdev = stdev_spread * (input_tensor.max() - input_tensor.min())

    gradients = torch.zeros_like(input_tensor)

    for i in range(num_samples):
        # Add noise
        noise = torch.randn_like(input_tensor) * stdev
        noisy_input = input_tensor + noise
        noisy_input = torch.clamp(noisy_input, 0, 1)
        noisy_input.requires_grad = True

        # Forward pass
        output = model(noisy_input)
        target_score = output[:, predicted_idx].sum()

        # Backward pass
        model.zero_grad()
        target_score.backward()

        # Accumulate gradients
        gradients += noisy_input.grad.data

        # Zero gradients
        noisy_input.grad.data.zero_()
        noisy_input.requires_grad = False

    # Average gradients
    avg_gradients = gradients / num_samples

    # Compute absolute values and sum over channels to get saliency map
    saliency = avg_gradients.abs().sum(dim=1).squeeze(0).cpu().numpy()

    # Normalize saliency map to [0, 1]
    saliency = (saliency - saliency.min()) / \
        (saliency.max() - saliency.min() + 1e-8)

    sgrad_importances = {}
    for segment_val in np.unique(segments):
        mask = (segments == segment_val)
        sgrad_importances[segment_val] = np.mean(saliency[mask])

    # Convert saliency map to 3-channel grayscale image
    saliency_image = np.stack([saliency, saliency, saliency], axis=-1)

    # Optional: Overlay saliency map onto the original image
    # Adjust the transparency with alpha if desired
    alpha = 1.0
    explanation_image = alpha * saliency_image + (1 - alpha) * input_image
    explanation_image = np.clip(explanation_image, 0, 1)

    # Convert to PIL Image
    explanation_image = Image.fromarray(
        (explanation_image * 255).astype(np.uint8))

    return saliency, explanation_image, sgrad_importances


def compare_explanations(lime_scores, sgrad_scores):
    # Ensure both score arrays have the same length
    assert len(lime_scores) == len(
        sgrad_scores), "Score arrays must be of the same length."

    # Rank the superpixels
    lime_ranks = np.argsort(lime_scores)[::-1]  # Descending order
    sgrad_ranks = np.argsort(sgrad_scores)[::-1]  # Descending order

    # To compute correlation, map the ranks
    # Create rank dictionaries
    lime_rank_dict = {superpixel: rank for rank,
                      superpixel in enumerate(lime_ranks)}
    sgrad_rank_dict = {superpixel: rank for rank,
                       superpixel in enumerate(sgrad_ranks)}

    # Align the ranks
    aligned_lime = []
    aligned_sgrad = []
    for superpixel in range(len(lime_scores)):
        aligned_lime.append(lime_rank_dict[superpixel])
        aligned_sgrad.append(sgrad_rank_dict[superpixel])

    # Compute Kendall-Tau
    tau, _ = kendalltau(aligned_lime, aligned_sgrad)

    # Compute Spearman Rank Correlation
    spearman, _ = spearmanr(aligned_lime, aligned_sgrad)

    return tau, spearman


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ResNet18 model
model = models.resnet18(pretrained=True)
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Define the image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the ImageNet class index mapping
with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
idx2synset = [class_idx[str(k)][0] for k in range(len(class_idx))]

imagenet_path = './imagenet_samples'
output_path = './output'
explanations_path = './explanations'
adversarial_explanations_path = './adversarial_explanations'

os.makedirs(output_path, exist_ok=True)
os.makedirs(explanations_path, exist_ok=True)
os.makedirs(adversarial_explanations_path, exist_ok=True)

# List of image file paths
image_paths = os.listdir(imagenet_path)

# Define PGD attack parameters
eps_list = [2/255, 4/255, 8/255]
steps_of_pgd = 5

# Initialize lists to store correlation results
kendall_taus = []
spearman_rhos = []


for eps in eps_list:
    print(f"\n--- Performing PGD Attack with Epsilon: {eps} ---\n")
    for img_filename in image_paths:
        # Construct the full image path
        img_path = os.path.join(imagenet_path, img_filename)

        print(f"Processing image: {img_path}")

        try:
            # Open and preprocess the image
            input_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}\n")
            continue

        input_tensor = preprocess(input_image)

        # Create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0).to(device)

        # Perform inference to get the original prediction
        with torch.no_grad():
            output = model(input_batch)

        # Get the predicted class index
        _, predicted_idx = torch.max(output, 1)
        predicted_idx = predicted_idx.item()
        predicted_synset = idx2synset[predicted_idx]
        predicted_label = idx2label[predicted_idx]

        print(
            f"Original Predicted Label: {predicted_synset} ({predicted_label})")

        # Define labels for attack (using predicted labels as targets)
        labels = torch.tensor([predicted_idx]).to(device)

        # Perform inference to get the original explanation
        input_batch.requires_grad = True
        output_orig = model(input_batch)
        model.zero_grad()
        loss_expl_orig = output_orig.gather(1, labels.unsqueeze(1)).squeeze()
        loss_expl_orig.backward()
        original_explanations = input_batch.grad.data.clone()
        input_batch.grad.data.zero_()
        input_batch.requires_grad = False

        # Perform PGD attack with the modified loss
        adv_img = pgd_attack(
            model,
            input_batch,
            labels,
            original_explanations,
            eps=eps,
            alpha=eps/steps_of_pgd,
            iters=steps_of_pgd,
            beta=0.5  # Adjust beta as needed
        )

        # Perform inference on adversarial images
        with torch.no_grad():
            output_adv = model(adv_img)

        # Get the predicted class index for adversarial image
        _, adv_predicted_idx = torch.max(output_adv, 1)
        adv_predicted_idx = adv_predicted_idx.item()
        adv_predicted_synset = idx2synset[adv_predicted_idx]
        adv_predicted_label = idx2label[adv_predicted_idx]

        print(
            f"Adversarial Predicted Label: {adv_predicted_synset} ({adv_predicted_label})\n")

        # Save the adversarial image
        # Remove file extension from original filename
        img_base_name = os.path.splitext(img_filename)[0]
        # Convert epsilon to integer (e.g., 2/255 -> 2)
        eps_scaled = int(eps * 255)
        # Construct the output filename
        output_filename = f"{img_base_name}_{eps_scaled}.jpeg"
        output_filepath = os.path.join(output_path, output_filename)

        # Convert the adversarial tensor to PIL Image
        # Undo the normalization
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        adv_img_denormalized = inv_normalize(adv_img.squeeze(0).cpu())
        adv_img_denormalized = torch.clamp(adv_img_denormalized, 0, 1)

        # Convert tensor to PIL Image
        adv_pil = transforms.ToPILImage()(adv_img_denormalized)

        try:
            # Save the image
            adv_pil.save(output_filepath)
            print(f"Saved adversarial image to: {output_filepath}\n")
        except Exception as e:
            print(f"Error saving adversarial image {output_filepath}: {e}\n")
            continue

        # Perform LIME explanation
        lime_weights, segments, lime_output_image = lime_explain(
            input_tensor, model, predicted_idx)

        # Perform Smoothed Gradients explanation
        saliency, sgrad_output_image, sgrad_importances = smoothed_gradients(
            input_tensor, model, predicted_idx, segments)

        # Prepare adv_img for smoothed_gradients
        adv_img_for_sgrad = adv_img.clone().detach()

        # Prepare adversarial image for smoothed_gradients
        adv_img_for_sgrad = adv_img.clone().detach()

        # Perform Smoothed Gradients explanation on adversarial image
        _, sgrad_output_image_adv, _ = smoothed_gradients(
            adv_img_for_sgrad, model, adv_predicted_idx, segments)

        # only save smooth grad and lime the first time, the rest don't need
        # to be saved only used for comparison.

        # save the smooth grad output image to folder using eps
        if eps == eps_list[0]:
            output_filename = f"{img_base_name}_sgrad.jpeg"
            output_filepath = os.path.join(explanations_path, output_filename)
            sgrad_output_image.save(output_filepath)

            output_filename = f"{img_base_name}_lime.jpeg"
            output_filepath = os.path.join(explanations_path, output_filename)
            lime_output_image.save(output_filepath)

        output_filename = f"{img_base_name}_{eps_scaled}_sgrad_adv.jpeg"
        output_filepath = os.path.join(
            adversarial_explanations_path, output_filename)
        sgrad_output_image_adv.save(output_filepath)

        superpixel_labels = np.unique(segments)
        superpixel_labels.sort()

        # Extract LIME importance scores
        lime_scores = np.array([lime_weights.get(int(sp), 0)
                               for sp in superpixel_labels])

        # Extract Smoothed Gradients importance scores
        sgrad_scores = np.array([sgrad_importances.get(
            int(sp), 0) for sp in superpixel_labels])

        # Only compute correlations once.
        if eps == eps_list[0]:
            try:
                tau, spearman = compare_explanations(lime_scores, sgrad_scores)
                print(f"Kendall-Tau Correlation: {tau:.4f}")
                print(f"Spearman Rank Correlation: {spearman:.4f}\n")

                # Store correlation results
                kendall_taus.append(tau)
                spearman_rhos.append(spearman)
            except Exception as e:
                print(
                    f"Error comparing explanations for image {img_path}: {e}\n")
                continue

# dump the correlation results to a file
correlation_results = {
    "kendall_taus": kendall_taus,
    "spearman_rhos": spearman_rhos
}

correlation_results_file = os.path.join(
    output_path, "correlation_results.json")

with open(correlation_results_file, "w") as f:
    json.dump(correlation_results, f)
