import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR

from methods.dgab import early_stopper
from methods.dgab.dgab_lpa import load_lpa_subtensor

from .dgab_model import GraphAttnModel


def dgab_main(feat_df, graph, train_idx, test_idx, labels, args, cat_features):
    device = args["device"]
    graph = graph.to(device)

    oof_predictions = torch.from_numpy(np.zeros([len(feat_df), 2])).float().to(device)
    test_predictions = torch.from_numpy(np.zeros([len(feat_df), 2])).float().to(device)

    kfold = StratifiedKFold(
        n_splits=args["n_fold"], shuffle=True, random_state=args["seed"]
    )

    y_target = labels.iloc[train_idx].values
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {
        col: torch.from_numpy(feat_df[col].values).long().to(device)
        for col in cat_features
    }

    y = labels
    labels = torch.from_numpy(y.values).long().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    best_fold_loss = float("inf")
    best_fold_model = None
    b_auc = 0.0
    b_ap = 0.0
    b_f1 = 0.0
    fold_metrics = {"auc": [], "ap": [], "f1": []}

    for fold, (trn_idx, val_idx) in enumerate(
        kfold.split(feat_df.iloc[train_idx], y_target)
    ):
        if fold >= 1:
            break
        print(f"Training fold {fold + 1}")
        trn_ind, val_ind = (
            torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(device),
            torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device),
        )

        sampler = MultiLayerFullNeighborSampler(args["n_layers"])
        train_dataloader = DataLoader(
            graph,
            trn_ind,
            sampler,
            device=device,
            batch_size=args["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        val_dataloader = DataLoader(
            graph,
            val_ind,
            sampler,
            device=device,
            batch_size=args["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        model = GraphAttnModel(
            in_feats=feat_df.shape[1],
            hidden_dim=args["hid_dim"] // 4,
            n_classes=2,
            heads=[4] * args["n_layers"],
            activation=nn.PReLU(),
            n_layers=args["n_layers"],
            drop=args["dropout"],
            device=device,
            gated=args["gated"],
            ref_df=feat_df,
            cat_features=cat_feat,
            n_bins=args["n_bins"],
            d_bin_dim=args["d_bin_dim"],
        ).to(device)

        if model.dynamic_binning_module is not None:
            train_features_np = feat_df.iloc[np.array(train_idx)[trn_idx]].values
            train_labels_np = y.iloc[np.array(train_idx)[trn_idx]].values
            x_init = torch.from_numpy(train_features_np).float()
            y_init = torch.from_numpy(train_labels_np).long()
            model.dynamic_binning_module.init_params(x_init, y_init)
            model = model.to(device)

        lr = args["lr"] * np.sqrt(args["batch_size"] / 1024)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args["wd"])
        steps_per_epoch = len(train_dataloader)
        total_steps = steps_per_epoch * args["max_epochs"]

        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=100,
        )

        earlystoper = early_stopper(
            patience=args["early_stopping"], verbose=True, mode="max"
        )
        global_step = 0
        for epoch in range(args["max_epochs"]):
            train_loss_list = []
            model.train()
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                global_step += 1
                batch_inputs, batch_work_inputs, batch_labels, lpa_labels = (
                    load_lpa_subtensor(
                        num_feat, cat_feat, labels, seeds, input_nodes, device
                    )
                )
                blocks = [block.to(device) for block in blocks]

                train_batch_logits = model(
                    blocks=blocks,
                    features=batch_inputs,
                    labels=lpa_labels,
                    n2v_feat=batch_work_inputs,
                )
                mask = batch_labels == 2
                train_batch_logits = train_batch_logits[~mask]
                batch_labels = batch_labels[~mask]

                train_loss = loss_fn(train_batch_logits, batch_labels)

                optimizer.zero_grad(set_to_none=True)
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss_list.append(train_loss.item())

                if step % 100 == 0:
                    with torch.no_grad():
                        tr_batch_pred = (
                            (torch.argmax(train_batch_logits, dim=1) == batch_labels)
                            .float()
                            .mean()
                        )
                        score = (
                            torch.softmax(train_batch_logits, dim=1)[:, 1].cpu().numpy()
                        )

                    try:
                        ___ = 1 + 1
                        print(
                            "In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, "
                            "train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}".format(
                                epoch,
                                step,
                                np.mean(train_loss_list) if train_loss_list else 0.0,
                                average_precision_score(
                                    batch_labels.cpu().numpy(), score
                                ),
                                tr_batch_pred.item(),
                                roc_auc_score(batch_labels.cpu().numpy(), score),
                            )
                        )
                    except:
                        pass

                del train_batch_logits, batch_inputs, batch_work_inputs, lpa_labels
                if step % 50 == 0:
                    torch.cuda.empty_cache()

            val_loss_sum = 0.0
            val_acc_sum = 0.0
            val_total_samples = 0
            val_all_scores = []
            val_all_labels = []
            model.eval()
            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                    batch_inputs, batch_work_inputs, batch_labels, lpa_labels = (
                        load_lpa_subtensor(
                            num_feat, cat_feat, labels, seeds, input_nodes, device
                        )
                    )

                    blocks = [block.to(device) for block in blocks]

                    val_batch_logits = model(
                        blocks=blocks,
                        features=batch_inputs,
                        labels=lpa_labels,
                        n2v_feat=batch_work_inputs,
                    )

                    oof_predictions[seeds] = val_batch_logits.detach()
                    mask = batch_labels == 2
                    val_batch_logits = val_batch_logits[~mask]
                    batch_labels = batch_labels[~mask]
                    batch_size = batch_labels.shape[0]
                    val_loss_sum += (
                        loss_fn(val_batch_logits, batch_labels).item() * batch_size
                    )
                    val_batch_pred = (
                        (torch.argmax(val_batch_logits, dim=1) == batch_labels)
                        .sum()
                        .item()
                    )
                    val_acc_sum += val_batch_pred
                    val_total_samples += batch_size
                    val_all_scores.append(
                        torch.softmax(val_batch_logits.detach(), dim=1)[:, 1]
                        .cpu()
                        .numpy()
                    )
                    val_all_labels.append(batch_labels.cpu().numpy())
                    if step % 100 == 0:
                        score = (
                            torch.softmax(val_batch_logits.detach(), dim=1)[:, 1]
                            .cpu()
                            .numpy()
                        )
                        try:
                            avg_loss = val_loss_sum / max(val_total_samples, 1)
                            avg_acc = val_acc_sum / max(val_total_samples, 1)
                            print(
                                "In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, "
                                "val_acc:{:.4f}, val_auc:{:.4f}".format(
                                    epoch,
                                    step,
                                    avg_loss,
                                    average_precision_score(
                                        batch_labels.cpu().numpy(), score
                                    ),
                                    avg_acc,
                                    roc_auc_score(batch_labels.cpu().numpy(), score),
                                )
                            )
                        except:
                            pass

            val_all_scores_cat = np.concatenate(val_all_scores)
            val_all_labels_cat = np.concatenate(val_all_labels)
            epoch_val_ap = average_precision_score(
                val_all_labels_cat, val_all_scores_cat
            )
            avg_val_loss = val_loss_sum / max(val_total_samples, 1)
            print(
                f"Epoch {epoch} summary - val_loss: {avg_val_loss:.6f}, "
                f"val_ap: {epoch_val_ap:.4f}, val_acc: {val_acc_sum / max(val_total_samples, 1):.4f}"
            )
            earlystoper.earlystop(epoch_val_ap, model, epoch=epoch)
            if earlystoper.is_earlystop:
                print(
                    f"Early Stopping! Best AP: {earlystoper._raw_best():.4f} at epoch {earlystoper.best_epoch}"
                )
                break
            del val_batch_logits, batch_inputs, batch_work_inputs, lpa_labels, blocks
            torch.cuda.empty_cache()

        test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
        test_sampler = MultiLayerFullNeighborSampler(args["n_layers"])
        test_dataloader = DataLoader(
            graph,
            test_ind,
            test_sampler,
            device=device,
            batch_size=args["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        b_model = earlystoper.get_best_model(device)
        b_model.eval()
        with torch.no_grad():
            fold_auc = 0.0
            fold_ap = 0.0
            fold_f1 = 0.0
            for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
                batch_inputs, batch_work_inputs, batch_labels, lpa_labels = (
                    load_lpa_subtensor(
                        num_feat, cat_feat, labels, seeds, input_nodes, device
                    )
                )

                blocks = [block.to(device) for block in blocks]

                test_batch_logits = b_model(
                    blocks=blocks,
                    features=batch_inputs,
                    labels=lpa_labels,
                    n2v_feat=batch_work_inputs,
                )

                test_predictions[seeds] = test_batch_logits.detach()
                if step % 10 == 0:
                    print("In test batch:{:04d}".format(step))
                del (
                    test_batch_logits,
                    batch_inputs,
                    batch_work_inputs,
                    lpa_labels,
                    blocks,
                )
                if step % 50 == 0:
                    torch.cuda.empty_cache()

            fold_test_score = (
                torch.softmax(test_predictions, dim=1)[test_idx, 1].cpu().numpy()
            )
            fold_y_target = labels[test_idx].cpu().numpy()
            fold_test_pred = (
                torch.argmax(test_predictions, dim=1)[test_idx].cpu().numpy()
            )

            fold_mask = fold_y_target != 2
            fold_test_score_filtered = fold_test_score[fold_mask]
            fold_y_target_filtered = fold_y_target[fold_mask]
            fold_test_pred_filtered = fold_test_pred[fold_mask]

            fold_auc = roc_auc_score(fold_y_target_filtered, fold_test_score_filtered)
            fold_ap = average_precision_score(
                fold_y_target_filtered, fold_test_score_filtered
            )
            fold_f1 = f1_score(
                fold_y_target_filtered, fold_test_pred_filtered, average="macro"
            )

            fold_metrics["auc"].append(fold_auc)
            fold_metrics["ap"].append(fold_ap)
            fold_metrics["f1"].append(fold_f1)
            if fold_auc > b_auc:
                b_auc = fold_auc
                b_ap = fold_ap
                b_f1 = fold_f1
            print(
                f"Fold {fold + 1} - AUC: {fold_auc:.4f}, AP: {fold_ap:.4f}, F1: {fold_f1:.4f}"
            )

        torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("Average metrics across all folds:")
    print(
        f"  AUC: {np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}"
    )
    print(
        f"  AP:  {np.mean(fold_metrics['ap']):.4f} ± {np.std(fold_metrics['ap']):.4f}"
    )
    print(
        f"  F1:  {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}"
    )
    print("=" * 50 + "\n")

    return b_auc, b_ap, b_f1
