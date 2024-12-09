import gc
import os
from typing import Optional
import torch
import torchvision
import torchvision.models as models

import tqdm
import pandas as pd

from model_mtd.model_mtd import MTDModel

from functools import partial

def get_imagenet12_ds(transform: Optional[callable] = None, batch_size: int = 32, shuffle: bool = True, num_workers: int = 16):
    imagenet_path = '/home/ran/datasets/imagenet12'

    def _default_transform(img):
        if not isinstance(img, torch.Tensor):
            img = torchvision.transforms.ToTensor()(img)

        return img

    if transform is None:
        transform = _default_transform

    imagenet_data = torchvision.datasets.ImageNet(imagenet_path, split='val', transform=transform)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers)
    
    return data_loader

def eval_torch_model(model, ds, use_cuda:bool = True, transform: Optional[callable] = None, n_subset: Optional[int] = None):
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available and use_cuda else "cpu")

    model.to(device)
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(ds):
            if n_subset and i>=n_subset:
                break

            if transform:
                images = transform(images)
            
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass through the model
            outputs = model(images)
            
            # Get the predicted classes
            _, predicted_top1 = torch.max(outputs.data, 1)
            _, predicted_top5 = torch.topk(outputs, 5, dim=1)
            
            # Update the accuracy metrics
            total += labels.size(0)
            correct_top1 += (predicted_top1 == labels).sum().item()
            correct_top5 += (labels.unsqueeze(1) == predicted_top5).any(1).sum().item()

    # Calculate and print the overall accuracy
    accuracy_top1 = 100 * correct_top1 / total
    accuracy_top5 = 100 * correct_top5 / total

    return accuracy_top1, accuracy_top5

if __name__ == '__main__':
    model_names = [
        'alexnet',
        'resnet18',
        'convnext_base',
        'densenet169',
    ]

    results = []

    for model_name in tqdm.tqdm(model_names):
        weights = models.get_model_weights(model_name).IMAGENET1K_V1
        preprocess = weights.transforms()

        ds = get_imagenet12_ds(transform=preprocess)
        model = models.get_model(model_name, weights=weights)

        mtd_model = MTDModel(model)
        eval_func = partial(eval_torch_model, ds=ds)

        accuray_top1_initial, accuracy_top5_initial = eval_func(mtd_model.model)
        results.append({
            'model_name': model_name,
            'model_state': 'initial',
            'ds': 'imagenet12_val',
            'acc_top1': accuray_top1_initial,
            'acc_top5': accuracy_top5_initial,
        })

        mtd_model.obfuscate_model(seed=1)
        accuray_top1_after_obf, accuracy_top5_after_obf = eval_func(mtd_model.model)

        results.append({
            'model_name': model_name,
            'model_state': 'after_obfuscation',
            'ds': 'imagenet12_val',
            'acc_top1': accuray_top1_after_obf,
            'acc_top5': accuracy_top5_after_obf,
        })

        mtd_model.deobfuscate_model()
        accuray_top1_after_deobf, accuracy_top5_after_deobf = eval_func(mtd_model.model)

        results.append({
            'model_name': model_name,
            'model_state': 'after_deobfuscation',
            'ds': 'imagenet12_val',
            'acc_top1': accuray_top1_after_deobf,
            'acc_top5': accuracy_top5_after_deobf,
        })

        del weights, preprocess, ds, model
        del mtd_model, eval_func
        
        gc.collect()

    df = pd.DataFrame(results)

    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)

    df.to_csv(os.path.join(results_dir, 'mtd_model_eval_results.csv'), index=False)

    

# if __name__ == '__main__':
#     weights = models.ResNet18_Weights.IMAGENET1K_V1
#     preprocess = weights.transforms()

#     ds = get_imagenet12_ds(transform=preprocess)

#     model = models.resnet18(weights=weights)
#     mtd_model = MTDModel(model)

#     eval_func = partial(eval_torch_model, ds=ds)

#     accuray_top1_initial, accuracy_top5_initial = eval_func(mtd_model.model)

#     print(f'Initial:')
#     print(f'\tTop1: {accuray_top1_initial:.2f}%')
#     print(f'\tTop5: {accuracy_top5_initial:.2f}%')

#     mtd_model.obfuscate_model(seed=1)
#     accuray_top1_after_obf, accuracy_top5_after_obf = eval_func(mtd_model.model)

#     print(f'After obfuscation:')
#     print(f'\tTop1: {accuray_top1_after_obf:.2f}%')
#     print(f'\tTop5: {accuracy_top5_after_obf:.2f}%')

#     mtd_model.deobfuscate_model()
#     accuray_top1_after_deobf, accuracy_top5_after_deobf = eval_func(mtd_model.model)

#     print(f'After deobfuscation:')
#     print(f'\tTop1: {accuray_top1_after_deobf:.2f}%')
#     print(f'\tTop5: {accuracy_top5_after_deobf:.2f}%')

    