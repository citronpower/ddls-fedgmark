from torch_geometric.datasets import TUDataset
import os

def convert_to_fedgmark_format(name):
    dataset = TUDataset(root=f'dataset/{name}', name=name)
    save_path = f'dataset/{name}/{name}.txt'
    os.makedirs(f'dataset/{name}', exist_ok=True)
    print(f"{name}: {dataset.num_classes} classes")

    with open(save_path, 'w') as f:
        f.write(f"{len(dataset)}\n")
        for data in dataset:
            num_nodes = data.num_nodes
            label = int(data.y.item())  # Graph label
            f.write(f"{num_nodes} {label}\n")

            edge_index = data.edge_index
            neighbors = [[] for _ in range(num_nodes)]
            for src, dst in edge_index.t().tolist():
                neighbors[src].append(dst)

            for i in range(num_nodes):
                if data.x is not None:
                    node_label = int(data.x[i].nonzero(as_tuple=True)[0][0]) if data.x[i].sum() > 0 else 0
                else:
                    node_label = 0  # Default node label for COLLAB
                neighs = neighbors[i]
                line = f"{len(neighs)} {node_label} " + " ".join(map(str, neighs)) + "\n"
                f.write(line)

    print(f"Converted {name} to FedGMark format at {save_path}")

# Convert DD and PROTEINS
convert_to_fedgmark_format("DD")
convert_to_fedgmark_format("PROTEINS")
convert_to_fedgmark_format("COLLAB")
