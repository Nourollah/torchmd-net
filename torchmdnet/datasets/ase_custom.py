import typing
import os
import pathlib
import warnings
import torch
from torch_geometric.data import Data, Dataset
from ase import io
from rich.progress import track


class ASECustom(Dataset):
    def __init__(self, path):
        """
        Args:
            path (str): Path to the .traj or .xyz or .xyzext file containing atomic structures.
        """
        super().__init__(path, transform=None, pre_transform=None)
        self.paths: typing.List[pathlib.Path] = []

        for i in os.listdir(path):
            try:
                if i.endswith('.traj') or i.endswith('.xyz') or i.endswith('.xyzext'):
                    self.paths.append(pathlib.Path(os.path.join(path, i)))
            except Exception as err:
                warnings.warn(f"{err}, file {i} is not a valid ASE file. Skipping this file.")
        print(f"Found {len(self.paths)} valid ASE files.")
        self.data_list: typing.List[Data] = []
        self._load_dataset()

    def _load_dataset(self):
        """
        Load the dataset from the specified path.
        This method reads the atomic structures from the file and returns a list of Data objects.
        """
        for file in track(self.paths):
            try:
                structure = io.read(file, index=':')
            except Exception as err:
                warnings.warn(f"{err} occurred while reading {file}. Skipping this file.")
                continue
            for i, _ in enumerate(structure):
                # Convert ASE Atoms object to PyG Data object
                pos = torch.tensor(structure[i].positions, dtype=torch.float)
                z = torch.tensor(structure[i].numbers, dtype=torch.long)
                energy = torch.tensor([structure[i].get_potential_energy()], dtype=torch.float)
                forces = torch.tensor(structure[i].get_forces(), dtype=torch.float)

                # Create edge index (optional: adjust threshold based on your requirements)
                edge_index = self._get_edge_index(pos)
                self.data_list.append(Data(z=z, pos=pos, edge_index=edge_index, y=energy, neg_dy=forces))



    def _get_edge_index(self, positions, threshold=1.6):
        """Generate edge index based on distance threshold."""
        import itertools
        edge_index = []
        for i, j in itertools.combinations(range(len(positions)), 2):
            dist = ((torch.tensor(positions[i]) - torch.tensor(positions[j])) ** 2).sum().sqrt()
            if dist < threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])  # Undirected graph

        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]



if __name__ == "__main__":
    # Test the dataset class with and without sampling
    dataset_root = '/home/masoud/Public'  # Replace with the actual dataset path

    # Without sampling
    dataset_no_sampling = ASECustom(path=dataset_root)
    print(f"Dataset loaded without sampling: {len(dataset_no_sampling)} samples.")
