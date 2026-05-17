import argparse
import os
from src.datasets import ChesapeakeRSC
from src.modules import CustomSemanticSegmentationTask
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import distance_transform_edt

def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_fn", required=True, type=str, help="Model checkpoint to load"
    )
    parser.add_argument(
        "--three_class", action="store_true", help="Whether to use three classes metrics"
    )
    parser.add_argument(
        "--cpu", default=0, type=int, help="GPU to use for inference (default: 0)"
    )
    parser.add_argument(
        "--eval_set", default="test", type=str, choices=["test", "val"], help="Which set to run over"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Whether to use TQDM progress bar"
    )
    return parser


def preprocess(sample):
    sample["image"] = sample["image"].float() / 255.0
    return sample


def main(args):
    model_fn = os.path.realpath(args.model_fn)
    assert os.path.exists(model_fn)

    device = torch.device("cpu")

    ds = ChesapeakeRSC("D:/run_forest/data/spatial-reasoning/", split=args.eval_set,
                       differentiate_tree_canopy_over_roads=True, transforms=preprocess)
    # ds = torch.utils.data.Subset(ds, range(1000))
    dl = DataLoader(ds, batch_size=8, num_workers=6)
    if not args.quiet:
        dl = tqdm(dl)

    task = CustomSemanticSegmentationTask.load_from_checkpoint(model_fn, map_location="cpu")
    model = task.model.eval().to(device)

    if args.three_class:
        cnf = np.zeros((3, 3), dtype=np.int64)

        for batch in dl:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with torch.inference_mode():
                preds = model(images).argmax(dim=1)

            for true_class_idx in [0, 1, 2]:
                true_mask = masks == true_class_idx
                for pred_class_idx in [0, 1, 2]:
                    pred_mask = preds == pred_class_idx
                    cnf[true_class_idx, pred_class_idx] += (true_mask & pred_mask).sum().item()

        # compute per class precision and recall from cnf
        recall_background = cnf[0, 0] / (cnf[0, 0] + cnf[0, 1] + cnf[0, 2])
        recall_road = cnf[1, 1] / (cnf[1, 0] + cnf[1, 1] + cnf[1, 2])
        recall_tree_canopy_over_road = cnf[2, 2] / (cnf[2, 0] + cnf[2, 1] + cnf[2, 2])

        precision_background = cnf[0, 0] / (cnf[0, 0] + cnf[1, 0] + cnf[2, 0])
        precision_road = cnf[1, 1] / (cnf[0, 1] + cnf[1, 1] + cnf[2, 1])
        precision_tree_canopy_over_road = cnf[2, 2] / (cnf[0, 2] + cnf[1, 2] + cnf[2, 2])

        print(
            f"{recall_background},{precision_background},{recall_road},{precision_road},{recall_tree_canopy_over_road},{precision_tree_canopy_over_road}")
    else:
        cnf = np.zeros((3, 2), dtype=np.int64)
        weighted_tp = 0.0
        weighted_total = 0.0
        for batch in dl:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with torch.inference_mode():
                preds = model(images).argmax(dim=1)

            for true_class_idx in [0, 1, 2]:
                true_mask = masks == true_class_idx
                for pred_class_idx in [0, 1]:
                    pred_mask = preds == pred_class_idx
                    cnf[true_class_idx, pred_class_idx] += (true_mask & pred_mask).sum().item()

            true_canopy_mask = (masks == 2).cpu().numpy()
            pred_road_mask = (preds == 1).cpu().numpy()
            visible_road_mask = (masks == 1).cpu().numpy()
            for b in range(masks.shape[0]):
                true_canopy_b = true_canopy_mask[b]
                pred_road_b = pred_road_mask[b]
                visible_road_b = visible_road_mask[b]
                if not np.any(true_canopy_b):
                    continue
                dist = distance_transform_edt(~visible_road_b)
                weights = dist[true_canopy_b]
                tp_pixels = true_canopy_b & pred_road_b
                weighted_tp += np.sum(weights[tp_pixels[true_canopy_b]])
                weighted_total += np.sum(weights)
        dwr = weighted_tp / weighted_total if weighted_total > 0 else 0.0
        recall_background = cnf[0, 0] / (cnf[0, 0] + cnf[0, 1])
        recall_road = cnf[1, 1] / (cnf[1, 0] + cnf[1, 1])
        recall_tree_canopy_over_road = cnf[2, 1] / (cnf[2, 0] + cnf[2, 1])

        precision_background = cnf[0, 0] / (cnf[0, 0] + cnf[1, 0] + cnf[2, 0])
        precision_road = cnf[1, 1] / (cnf[0, 1] + cnf[1, 1] + cnf[2, 1])
        precision_tree_canopy_over_road = cnf[2, 1] / (cnf[0, 1] + cnf[1, 1] + cnf[2, 1]) if (cnf[0, 1] + cnf[1, 1] +
                                                                                              cnf[2, 1]) > 0 else 0.0

        f1_road = 2 * precision_road * recall_road / (precision_road + recall_road) if (precision_road + recall_road) > 0 else 0.0
        f1_tree_canopy_over_road = 2 * precision_tree_canopy_over_road * recall_tree_canopy_over_road / (
                precision_tree_canopy_over_road + recall_tree_canopy_over_road) if (precision_tree_canopy_over_road + recall_tree_canopy_over_road) > 0 else 0.0
        print(
            f"road: recall = {recall_road}, precision = {precision_road}, f1-score = {f1_road}\n"
            f"tree_canopy_over_road: recall = {recall_tree_canopy_over_road}, precision = {precision_tree_canopy_over_road}, f1-score = {f1_tree_canopy_over_road}")
        print(f"DWR: {dwr:.4f}")

        cm_2class = np.zeros((2, 2), dtype=np.int64)
        cm_2class[0, 0] = cnf[0, 0]  # TP фон
        cm_2class[0, 1] = cnf[0, 1]  # фон → дороги (FP для фона)
        cm_2class[1, 0] = cnf[1, 0] + cnf[2, 0]  # дороги → фон (FN для дорог)
        cm_2class[1, 1] = cnf[1, 1] + cnf[2, 1]  # TP дороги (открытые + скрытые)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_2class, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Фон', 'Дороги'],
                    yticklabels=['Фон (true)', 'Дороги (true)'])
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        plt.title('Матрица ошибок (2-классовый режим)')
        plt.tight_layout()
        plt.savefig('confusion_matrix_2class.png', dpi=300)
        print("Матрица ошибок сохранена как confusion_matrix_2class.png")


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    main(args)
