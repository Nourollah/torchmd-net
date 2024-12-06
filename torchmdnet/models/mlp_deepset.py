from typing import Callable, Dict, Optional, Union, List
import typing
import rich

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from torch.nn import init

from torchmdnet.models.utils import (
    NeighborEmbedding,
    CosineCutoff,
    OptimizedDistance,
    rbf_class_mapping,
    act_class_mapping,
    scatter,
)

# from src import utils
# from src.properties import properties

__all__ = ["MLPDeepSet"]



ORBITALS = "1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p 7s 5f 6d 7p 6f 7d 7f".split()
POSSIBLE_ELECTRONS = dict(s=2, p=6, d=10, f=14)


def generate_electron_configurations(atomic_num: int) -> typing.List[int]:
    """
    Generate electron configuration for a given atomic number.

    :param atomic_num: atomic number
    :return: electron configuration
    """
    electron_count, last_idx, config = 0, -1, []
    for i in ORBITALS:
        if electron_count < atomic_num:
            config.append(POSSIBLE_ELECTRONS[i[-1]])
            electron_count += POSSIBLE_ELECTRONS[i[-1]]
            last_idx += 1
        else:
            config.append(0)
    if electron_count > atomic_num:
        config[last_idx] -= electron_count - atomic_num
    return config


# class Transform(torch.nn.Module):
#     """
#     Base class for all transforms.
#     The base class ensures that the reference to the data and datamodule attributes are
#     initialized.
#     Transforms can be used as pre- or post-processing layers.
#     They can also be used for other parts of a model, that need to be
#     initialized based on data.

#     To implement a new transform, override the forward method. Preprocessors are applied
#     to single examples, while postprocessors operate on batches. All transforms should
#     return a modified `inputs` dictionary.

#     """

#     def datamodule(self, value):
#         """
#         Extract all required information from data module automatically when using
#         PyTorch Lightning integration. The transform should also implement a way to
#         set these things manually, to make it usable independent of PL.

#         Do not store the datamodule, as this does not work with torchscript conversion!
#         """
#         pass

#     def forward(
#         self,
#         inputs: Dict[str, torch.Tensor],
#     ) -> Dict[str, torch.Tensor]:
#         raise NotImplementedError

#     def teardown(self):
#         pass

def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)


def build_mlp(input_dim, hidden_dim, output_dim, num_layers, activation=nn.SiLU):
    """
    Function to build an MLP with a specified number of layers, where all hidden layers
    have the same dimensions and use a customizable activation function.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden layers.
        output_dim (int): Dimension of the output features.
        num_layers (int): Number of hidden layers in the model.
        activation (nn.Module): Activation function to use (default is nn.SiLU).

    Returns:
        nn.Sequential: The MLP model.
    """
    layers = [nn.Linear(input_dim, hidden_dim), activation()]  # First layer and activation

    # Add hidden layers with the same dimensions
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), activation()])

    # Add the final output layer (no activation for output layer)
    layers.append(nn.Linear(hidden_dim, output_dim))

    # Return the model as a Sequential module
    return nn.Sequential(*layers)



class CollectAtomTriples(torch.nn.Module):

    def forward(
        self,
        idx_i: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Using the neighbors contained within the cutoff shell, generate all unique pairs
        of neighbors and convert them to index arrays. Applied to the neighbor arrays,
        these arrays generate the indices involved in the atom triples.

        Example:
            idx_j[idx_j_triples] -> j atom in triple
            idx_j[idx_k_triples] -> k atom in triple
            Rij[idx_j_triples] -> Rij vector in triple
            Rij[idx_k_triples] -> Rik vector in triple
        """

        _, n_neighbors = torch.unique_consecutive(idx_i, return_counts=True)

        offset = 0
        idx_i_triples = ()
        idx_jk_triples = ()
        for idx in range(n_neighbors.shape[0]):
            triples = torch.combinations(
                torch.arange(offset, offset + n_neighbors[idx]), r=2
            )
            idx_i_triples += (torch.ones(triples.shape[0], dtype=torch.long) * idx,)
            idx_jk_triples += (triples,)
            offset += n_neighbors[idx]

        idx_i_triples = torch.cat(idx_i_triples)

        idx_jk_triples = torch.cat(idx_jk_triples)
        idx_j_triples, idx_k_triples = idx_jk_triples.split(1, dim=-1)

        return idx_i_triples, idx_j_triples.squeeze(-1), idx_k_triples.squeeze(-1)

        # def teardown(self):
        #     pass

        # def datamodule(self, value):
        #     """
        #     Extract all required information from data module automatically when using
        #     PyTorch Lightning integration. The transform should also implement a way to
        #     set these things manually, to make it usable independent of PL.

        #     Do not store the datamodule, as this does not work with torchscript conversion!
        #     """
        #     pass

def scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    """
    Sum over values with the same indices.

    Args:
            x: input values
            idx_i: index of center atom i
            dim_size: size of the dimension after reduction
            dim: the dimension to reduce

    Returns:
            reduced input

    """
    return _scatter_add(x, idx_i, dim_size, dim)


@torch.jit.script
def _scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)
    return y


class PairAtomsDistanceAdumbration(torch.nn.Module):
    """Compute the distance between atoms."""

    def __init__(self,
                 orbitals_size: int = 22, ):
        """
        Args:
                cutoff: cutoff radius
        """
        super(PairAtomsDistanceAdumbration, self).__init__()
        self.orbitals_size = orbitals_size

    def forward(self,
                z: torch.Tensor,
                idx_i: torch.Tensor,
                idx_j: torch.Tensor,
                d_ij: torch.Tensor,
                phi_ij: torch.Tensor,
    ):
        """
        Create a representation of molecules based on the distance between atoms.

        Args:
                inputs:
                        - atoms (torch.Tensor): index and atomic numbers of atoms.
                        - r_ij (torch.Tensor): pairwise distance between atoms.
                        - idx_i (torch.Tensor): index of center atom i
                        - idx_j (torch.Tensor): index of neighbors j

        Returns:
                torch.Tensor: distance between atoms.
        """
        # Initialize the tensor
        representation: torch.Tensor = torch.zeros(d_ij.size(0),
                                                   2 * self.orbitals_size + 1 + phi_ij.size(-1),
                                                   device=d_ij.device)
        atoms_electron_config = torch.tensor(
            [generate_electron_configurations(i) for i in z.squeeze().tolist()],
            dtype=torch.float32,
            device=d_ij.device)
        representation[:, -1] = torch.squeeze(d_ij)
        representation[:, :self.orbitals_size] = atoms_electron_config[idx_i]
        representation[:, self.orbitals_size:2 * self.orbitals_size] = atoms_electron_config[idx_j]
        representation[:, 2 * self.orbitals_size:-1] = torch.squeeze(phi_ij)
        # return the representation
        return representation


class TripleAtomsDistanceAdumbration(torch.nn.Module):
    """Compute the distance between atoms."""

    def __init__(self,
                 orbitals_size: int = 22, ):
        """
        Args:
                cutoff: cutoff radius
        """
        super(TripleAtomsDistanceAdumbration, self).__init__()
        self.orbitals_size = orbitals_size

    def forward(self,
                triple_idx_i: torch.Tensor,
                triple_idx_j: torch.Tensor,
                triple_idx_k: torch.Tensor,
                idx_i: torch.Tensor,
                idx_j: torch.Tensor,
                z: torch.Tensor,
                positions: torch.Tensor):
        """
        Create a representation of molecules based on the distance between atoms.

        Args:
                inputs: The dictionary of input tensors.

        Returns:
                torch.Tensor: The triple total graph with all distances which are included.
        """

        atoms_electron_config = torch.tensor([generate_electron_configurations(i) for i in z.squeeze().tolist()],
                                             dtype=torch.float32,
                                             device=idx_j.device)

        # Initializing the zeros tensor to keep the atomic numbers and distances
        triple_representation = torch.zeros(triple_idx_i.size(0), self.orbitals_size * 3 + 3, device=idx_j.device)

        triple_representation[:, 0:self.orbitals_size] = atoms_electron_config[triple_idx_i]
        triple_representation[:, self.orbitals_size:2 * self.orbitals_size] = atoms_electron_config[idx_j[triple_idx_j]]
        triple_representation[:, 2 * self.orbitals_size:3 * self.orbitals_size] = atoms_electron_config[idx_j[triple_idx_k]]

        # Compute distances
        r_ij = torch.norm(positions[triple_idx_i] - positions[idx_j[triple_idx_j]], dim=1)
        r_ik = torch.norm(positions[triple_idx_i] - positions[idx_j[triple_idx_k]], dim=1)
        # r_jk = torch.norm(positions[idx_j[triple_idx_j]] - positions[idx_j[triple_idx_k]], dim=1)

        # Compute angles
        vec_ij = positions[idx_j[triple_idx_j]] - positions[triple_idx_i]
        vec_ik = positions[idx_j[triple_idx_k]] - positions[triple_idx_i]
        # vec_jk = positions[idx_j[triple_idx_k]] - positions[idx_j[triple_idx_j]]

        cos_theta_ijk = torch.sum(vec_ij * vec_ik, dim=1) / (r_ij * r_ik)
        # cos_theta_jki = torch.sum(-vec_ij * vec_jk, dim=1) / (r_ij * r_jk)

        # theta_ijk = torch.acos(cos_theta_ijk)
        # theta_jki = torch.acos(cos_theta_jki)

        # Store distances and angles in the representation
        triple_representation[:, -3] = r_ij
        triple_representation[:, -2] = r_ik
        # triple_representation[:, -3] = r_jk
        triple_representation[:, -1] = cos_theta_ijk
        # triple_representation[:, -1] = theta_jki

        return triple_representation


class MLPDeepSet(nn.Module):

    def __init__(
        self,
        n_atom_basis: int,
        base_cutoff: float,
        inner_cutoff: float,
        outer_cutoff: float,
        max_num_neighbors: int = 32,
        embedding_size: int = 256,
        # mlp_size: int = 20,
        radial_basis: nn.Module = GaussianRBF(20, 15.0),
        use_vector_representation: bool = False,
        forces_based_on_energy: bool = False,
        close_far_split: bool = True,
        using_triplet_module: bool = False,
    ):

        super(MLPDeepSet, self).__init__()

        self.n_atom_basis = n_atom_basis
        self.inner_cutoff = inner_cutoff
        self.inner_cutoff = outer_cutoff
        self.base_cutoff = base_cutoff
        self.radial_basis = radial_basis
        self.embedding_size = embedding_size

        self.use_vector_representation = use_vector_representation
        self.forces_based_on_energy = forces_based_on_energy
        self.close_far_split = close_far_split
        self.using_triplet_module = using_triplet_module

        self.pair_atoms_coder = PairAtomsDistanceAdumbration()

        # self.distance = OptimizedDistance(
        #     base_cutoff,
        #     outer_cutoff,
        #     max_num_pairs=max_num_neighbors,
        #     return_vecs=True,
        #     loop=True,
        #     box=None,
        #     long_edge_index=True,
        #     check_errors=True, # Set False if there are more than 10k neighbors and it throw an error. Check this thread: https://github.com/torchmd/torchmd-net/issues/203
        # )

        try:
            # Initialize MLPs
            in_size: int = 45 + self.radial_basis.n_rbf
            if self.close_far_split:
                self.inner_scalar_mlp = build_mlp(in_size, embedding_size, embedding_size, 3, activation=nn.SiLU)

                self.outer_scalar_mlp = build_mlp(in_size, embedding_size, embedding_size, 3, activation=nn.SiLU)

                if self.use_vector_representation:
                    self.inner_vector_mlp = build_mlp(in_size, embedding_size, 3 * embedding_size, 3, activation=nn.SiLU)

                    self.outer_vector_mlp = build_mlp(in_size, embedding_size, 3 * embedding_size, 3, activation=nn.SiLU)

            else:
                self.scalar_mlp = build_mlp(in_size, embedding_size, embedding_size, 3, activation=nn.SiLU)
                if self.use_vector_representation:
                    self.vector_mlp = build_mlp(in_size, embedding_size, 3 * embedding_size, 3, activation=nn.SiLU)

            if self.using_triplet_module:
                self.triplet_atoms_coder = TripleAtomsDistanceAdumbration()
                self.triplet_scalar_processor_mlp = build_mlp(69, 3 * embedding_size, embedding_size, 3, activation=nn.SiLU)

                if self.use_vector_representation:
                    self.triplet_vector_processor_mlp = build_mlp(69, 3 * embedding_size, 3 * embedding_size, 3, activation=nn.SiLU)

        except Exception as e:
            print(f"Error initializing MLP layers: {e}")


    def reset_parameters(self):
        if self.close_far_split:
            for layer in self.inner_scalar_mlp:
	            if isinstance(layer, nn.Linear):
		            init.xavier_uniform_(layer.weight)
		            init.zeros_(layer.weight) if layer.bias is not None else None
                    # if layer.bias is not None:
                    #     init.zeros_(layer.weight)
            for layer in self.outer_scalar_mlp:
                if isinstance(layer, nn.Linear):
                    init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        init.zeros_(layer.weight)
            if self.use_vector_representation:
                for layer in self.inner_vector_mlp:
                    if isinstance(layer, nn.Linear):
                        init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            init.zeros_(layer.weight)
                for layer in self.outer_vector_mlp:
                    if isinstance(layer, nn.Linear):
                        init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            init.zeros_(layer.weight)
        else:
            for layer in self.scalar_mlp:
                if isinstance(layer, nn.Linear):
                    init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        init.zeros_(layer.weight)
            if self.use_vector_representation:
                for layer in self.vector_mlp:
                    if isinstance(layer, nn.Linear):
                        init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            init.zeros_(layer.weight)
        if self.using_triplet_module:
            for layer in self.triplet_scalar_processor_mlp:
                if isinstance(layer, nn.Linear):
                    init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        init.zeros_(layer.weight)
            if self.use_vector_representation:
                for layer in self.triplet_vector_processor_mlp:
                    if isinstance(layer, nn.Linear):
                        init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            init.zeros_(layer.weight)


    def forward(self,
                z: torch.Tensor,
                pos: torch.Tensor,
                batch: torch.Tensor,
                box: Optional[torch.Tensor] = None,
                q: Optional[torch.Tensor] = None,
                s: Optional[torch.Tensor] = None,
                extra_args: Optional[Dict[str, torch.Tensor]] = None) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_index, edge_weight, edge_vec = extra_args["computed_values"]["edge_index"], extra_args["computed_values"]["edge_weight"], extra_args["computed_values"]["edge_vec"]
        phi_ij = self.radial_basis(edge_weight)
        idx_i = edge_index[0, :]
        idx_j = edge_index[1, :]

        pair_atoms_repr = self.pair_atoms_coder(z, idx_i, idx_j, edge_weight, phi_ij)

        q = self.process_pair_scalar(pair_atoms_repr, idx_i)
        mu = self.process_pair_vector(pair_atoms_repr, idx_i) if self.use_vector_representation else None

        if self.using_triplet_module:
            idx_i_triples, idx_j_triples, idx_k_triples = extra_args["computed_values"]["idx_i_triples"], extra_args["computed_values"]["idx_j_triples"], extra_args["computed_values"]["idx_k_triples"]
            triplet_atoms_repr = self.triplet_atoms_coder(idx_i_triples, idx_j_triples, idx_k_triples, idx_i, idx_j, z, pos)
            tq = self.triplet_scalar_pass(triplet_atoms_repr, idx_i, idx_i_triples)
            tmu = self.triplet_vector_pass(triplet_atoms_repr, idx_i, idx_i_triples) if self.use_vector_representation else None

        return q, mu, z, pos, batch

    def process_pair_scalar(self, pair_atoms_repr: torch.Tensor, idx_i: torch.Tensor) -> torch.Tensor:
        if self.close_far_split:
            close_data_index = pair_atoms_repr[:, -1] < self.inner_cutoff
            far_data_index = pair_atoms_repr[:, -1] >= self.inner_cutoff
            return self.close_far_pair_scalar_pass(pair_atoms_repr, idx_i, close_data_index, far_data_index)
        else:
            return self.pair_scalar_pass(pair_atoms_repr, idx_i)

    def process_pair_vector(self, pair_atoms_repr: torch.Tensor, idx_i: torch.Tensor) -> torch.Tensor:
        if self.close_far_split:
            close_data_index = pair_atoms_repr[:, -1] < self.inner_cutoff
            far_data_index = pair_atoms_repr[:, -1] >= self.inner_cutoff
            return self.close_far_pair_vector_pass(pair_atoms_repr, idx_i, close_data_index, far_data_index)
        else:
            return self.pair_vector_pass(pair_atoms_repr, idx_i)

    def pair_scalar_pass(self, pair_atoms_repr: torch.Tensor, idx_i: torch.Tensor) -> torch.Tensor:
        q = self.scalar_mlp(pair_atoms_repr)
        counts = torch.bincount(idx_i)
        buf = torch.zeros((counts.size(0), self.embedding_size), dtype=q.dtype, device=q.device)
        q = buf.index_add(0, idx_i, q)
        return q

    def close_far_pair_scalar_pass(self, pair_atoms_repr: torch.Tensor, idx_i: torch.Tensor,
                                   close_data_index: torch.Tensor, far_data_index: torch.Tensor) -> torch.Tensor:
        q = torch.zeros((pair_atoms_repr.size(0), self.embedding_size), device=pair_atoms_repr.device, dtype=pair_atoms_repr.dtype)
        q[close_data_index] = self.inner_scalar_mlp(pair_atoms_repr[close_data_index])
        q[far_data_index] = self.outer_scalar_mlp(pair_atoms_repr[far_data_index])
        counts = torch.bincount(idx_i)
        buf = torch.zeros((counts.size(0), self.embedding_size), dtype=q.dtype, device=q.device)
        q = buf.index_add(0, idx_i, q)
        return q

    def pair_vector_pass(self, pair_atoms_repr: torch.Tensor, idx_i: torch.Tensor) -> torch.Tensor:
        mu = self.inner_vector_mlp(pair_atoms_repr).reshape(idx_i.size(0), 3, self.n_atom_basis)
        counts = torch.bincount(idx_i)
        buf = torch.zeros((counts.size(0), mu.size(1), mu.size(2)), dtype=mu.dtype, device=mu.device)
        mu = buf.index_add(0, idx_i, mu)
        return mu

    def close_far_pair_vector_pass(self, pair_atoms_repr: torch.Tensor, idx_i: torch.Tensor,
                                   close_data_index: torch.Tensor, far_data_index: torch.Tensor) -> torch.Tensor:
        mu = torch.zeros((idx_i.size(0), 3, self.embedding_size), device=pair_atoms_repr.device, dtype=pair_atoms_repr.dtype)
        mu[close_data_index] = self.inner_vector_mlp(pair_atoms_repr[close_data_index]).reshape(idx_i[close_data_index].size(0), 3, self.embedding_size)
        mu[far_data_index] = self.outer_vector_mlp(pair_atoms_repr[far_data_index]).reshape(idx_i[far_data_index].size(0), 3, self.embedding_size)
        counts = torch.bincount(idx_i)
        buf = torch.zeros((counts.size(0), mu.size(1), mu.size(2)), dtype=mu.dtype, device=mu.device)
        mu = buf.index_add(0, idx_i, mu)
        return mu

    def triplet_scalar_pass(self, triplet_atoms_repr: torch.Tensor, idx_i: torch.Tensor, triplet_idx_i: torch.Tensor) -> torch.Tensor:
        tq = self.triplet_scalar_processor_mlp(triplet_atoms_repr)
        count = torch.bincount(idx_i)
        buf = torch.zeros((count.size(0), self.embedding_size), dtype=tq.dtype, device=tq.device)
        tq = buf.index_add(0, triplet_idx_i, tq)
        return tq

    def triplet_vector_pass(self, triplet_atoms_repr: torch.Tensor, idx_i: torch.Tensor, triplet_idx_i: torch.Tensor) -> torch.Tensor:
        tmu = self.triplet_vector_processor_mlp(triplet_atoms_repr).reshape(triplet_atoms_repr.size(0), 3, self.embedding_size)
        count = torch.bincount(idx_i)
        buf = torch.zeros((count.size(0), tmu.size(1), self.embedding_size), dtype=tmu.dtype, device=tmu.device)
        tmu = buf.index_add(0, triplet_idx_i, tmu)
        return tmu


if __name__ == "__main__":

    import torch
    import numpy as np
    import random
    import time

    def set_seed(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set the seed
    set_seed(2000)
    # Test the model
    model = MLPDeepSet(n_atom_basis=22,
                base_cutoff=0.0,
                inner_cutoff=1.0,
                outer_cutoff=3.0,
                # radial_basis=None,
                use_vector_representation=True,
                forces_based_on_energy=False,
                close_far_split=True,
                using_triplet_module=True)
    # Generate random input
    z = torch.randint(1, 18, (100, 1))
    print(z.size())
    pos = torch.randint(0, 3, (18, 3), dtype=torch.float32)
    batch = torch.zeros(100, dtype=torch.long)
    # box = torch.rand(3, 3)
    edge_index, edge_weight, edge_vec = model(z, pos, batch)
    tt = CollectAtomTriples()

    start_time = time.time()

    zz = tt(edge_index[0, :])

    end_time = time.time()
    # Compute the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.6f} seconds")
    rich.print(edge_index)
    rich.print(edge_weight)
    rich.print(edge_vec)

