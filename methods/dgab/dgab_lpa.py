def load_lpa_subtensor(node_feat, work_node_feat, labels, seeds, input_nodes, device):
    batch_inputs = node_feat[input_nodes]
    batch_work_inputs = {
        i: work_node_feat[i][input_nodes] for i in work_node_feat if i not in {"Labels"}
    }
    batch_labels = labels[seeds]

    propagate_labels = labels[input_nodes].clone()
    propagate_labels[: seeds.shape[0]] = 2
    return batch_inputs, batch_work_inputs, batch_labels, propagate_labels
