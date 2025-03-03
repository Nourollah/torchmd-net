from typing import Callable, Dict, Optional, Union, List
import typing
import rich

import torch
import torch.nn as nn
import torch_geometric as tg
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


ELEMENTLOGPRIME = {'Na': 0.301029995663981, 'K': 0.477121254719662,
                   'H': 0.698970004336019, 'Rb': 0.845098040014257,
                   'Li': 1.04139268515823, 'Cs': 1.11394335230684,
                   'Fr': 1.23044892137827, 'Ca': 1.27875360095283,
                   'Mg': 1.36172783601759, 'Ba': 1.46239799789896,
                   'Sr': 1.49136169383427, 'Be': 1.568201724067,
                   'Ra': 1.61278385671974, 'Y': 1.63346845557959,
                   'Sc': 1.67209785793572, 'Lu': 1.72427586960079,
                   'Lr': 1.77085201164214, 'Ti': 1.78532983501077,
                   'Zr': 1.82607480270083, 'Hf': 1.85125834871908,
                   'Rf': 1.86332286012046, 'V': 1.89762709129044,
                   'Nb': 1.91907809237607, 'Ta': 1.94939000664491,
                   'Db': 1.98677173426624, 'Cr': 2.00432137378264,
                   'W': 2.01283722470517, 'Mo': 2.02938377768521,
                   'Sg': 2.03742649794062, 'Mn': 2.05307844348342,
                   'Re': 2.10380372095596, 'Tc': 2.11727129565576,
                   'Bh': 2.13672056715641, 'Fe': 2.14301480025409,
                   'Os': 2.17318626841227, 'Ru': 2.17897694729317,
                   'Hs': 2.19589965240923, 'Co': 2.21218760440396,
                   'Rh': 2.22271647114758, 'Ir': 2.2380461031288,
                   'Mt': 2.25285303097989, 'Ni': 2.25767857486918,
                   'Pd': 2.28103336724773, 'Pt': 2.28555730900777,
                   'Ds': 2.29446622616159, 'Cu': 2.29885307640971,
                   'Ag': 2.32428245529769, 'Au': 2.34830486304816,
                   'Rg': 2.35602585719312, 'Zn': 2.35983548233989,
                   'Cd': 2.36735592102602, 'Hg': 2.37839790094814,
                   'Cn': 2.38201704257487, 'Al': 2.39967372148104,
                   'Ga': 2.40993312333129, 'B': 2.41995574848976,
                   'Tl': 2.42975228000241, 'In': 2.43296929087441,
                   'Nh': 2.44247976906445, 'Si': 2.44870631990508,
                   'C': 2.45178643552429, 'Pb': 2.46686762035411,
                   'Sn': 2.48713837547719, 'Ge': 2.49276038902684,
                   'Fl': 2.49554433754645, 'P': 2.50105926221775,
                   'N': 2.51982799377572, 'As': 2.52762990087134,
                   'Sb': 2.54032947479087, 'Bi': 2.54282542695918,
                   'Mc': 2.54777470538782, 'O': 2.55509444857832,
                   'S': 2.56466606425209, 'Se': 2.57170883180869,
                   'Te': 2.57863920996807, 'Po': 2.58319877396862,
                   'Lv': 2.58994960132571, 'F': 2.59879050676312,
                   'Cl': 2.60314437262018, 'Br': 2.61172330800734,
                   'I': 2.6222140229663, 'At': 2.62428209583567,
                   'Ts': 2.63447727016073, 'Ar': 2.63648789635337,
                   'He': 2.64246452024212, 'Ne': 2.64640372622307,
                   'Kr': 2.65224634100332, 'Xe': 2.65991620006985,
                   'Rn': 2.66370092538965, 'Og': 2.66558099101795,
                   'Ce': 2.66931688056611, 'Nd': 2.68033551341456,
                   'La': 2.68752896121463, 'Th': 2.69108149212297,
                   'Pr': 2.69810054562339, 'Sm': 2.70156798505593,
                   'Gd': 2.70671778233676, 'Dy': 2.71683772329952,
                   'Er': 2.71850168886727, 'Yb': 2.73319726510657,
                   'U': 2.73798732633343, 'Eu': 2.74585519517373,
                   'Ho': 2.75050839485135, 'Tb': 2.75511226639507,
                   'Tm': 2.75663610824585, 'Pa': 2.76117581315573,
                   'Ac': 2.76863810124761, 'Pu': 2.77305469336426,
                   'Np': 2.77742682238931, 'Pm': 2.77887447200274,
                   'Am': 2.78318869107526, 'Cm': 2.78746047451841,
                   'Bk': 2.79028516403324, 'Cf': 2.79169064902012,
                   'Es': 2.80002935924413, 'Fm': 2.80685802951882,
                   'Md': 2.80821097292422, 'No': 2.8109042806687}

ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51,
    'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61,
    'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81,
    'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91,
    'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101,
    'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
    'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

# Convert the dictionary using atomic numbers as keys
Z_LOGPRIME = {ATOMIC_NUMBERS[element]: value for element, value in ELEMENTLOGPRIME.items()}


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


def CollectAtomTriplesExtended(connections, pos, cutoff):
    """
    Highly optimized triplet generation with distance cutoff.
    Considers neighbors of both i and j as potential k atoms.

    Args:
        connections (torch.Tensor): 2D tensor of atom connections (edges).
        pos (torch.Tensor): Positions of atoms (3D coordinates).
        cutoff (float): Distance threshold for filtering triplets.
    Returns:
        torch.Tensor: Tensor of valid triplets (i, j, k).
    """
    # Number of atoms
    num_atoms = pos.size(0)
    # Create an efficient adjacency list representation
    max_neighbors = torch.max(torch.bincount(connections[:, 0]))
    neighbor_indices = torch.full((num_atoms, max_neighbors.item() * 2), -1, dtype=torch.long, device=pos.device)
    neighbor_counts = torch.zeros(num_atoms, dtype=torch.long, device=pos.device)

    # Populate the adjacency list for both i and j
    for u, v in connections:
        # Add neighbors for both u and v
        for atom in [u, v]:
            for neighbor in [u, v]:
                if atom != neighbor:
                    idx = neighbor_counts[atom]
                    if idx < max_neighbors * 2:
                        neighbor_indices[atom, idx] = neighbor
                        neighbor_counts[atom] += 1

    # Preallocate memory for triplets (with a generous upper bound)
    max_possible_triplets = connections.size(0) * max_neighbors * 2
    triplets = torch.zeros((max_possible_triplets, 3), dtype=torch.long, device=pos.device)
    triplet_count = torch.tensor(0, dtype=torch.long, device=pos.device)

    # Kernel to generate triplets
    @torch.no_grad()
    def generate_triplets_kernel():
        for idx in range(connections.size(0)):
            i, j = connections[idx]

            # Iterate through neighbors of both i and j
            for k_idx in range(max_neighbors * 2):
                k = neighbor_indices[j, k_idx]

                # Break if no more neighbors or invalid neighbor
                if k == -1 or k == i or k == j:
                    continue

                # Check distance cutoffs for all three atoms
                if (torch.norm(pos[i] - pos[k]) < cutoff and
                    torch.norm(pos[j] - pos[k]) < cutoff):
                    # Atomic add to avoid race conditions
                    current_count = triplet_count.item()
                    triplets[current_count] = torch.tensor([i, j, k], device=pos.device)
                    triplet_count.add_(1)

    # Run the kernel
    generate_triplets_kernel()

    # Trim to actual number of triplets
    final_triplets = triplets[:triplet_count.item()]

    return final_triplets[:, 0], final_triplets[:, 1], final_triplets[:, 2]


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
	            x: torch.Tensor,
	            idx_i: torch.Tensor,
	            idx_j: torch.Tensor,
	            d_ij: torch.Tensor,
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
		                                           256 * 2 + 2 * self.orbitals_size + 1,
		                                           device=d_ij.device)
		atoms_electron_config = torch.tensor(
			[generate_electron_configurations(i) for i in z.squeeze().tolist()],
			dtype=torch.float32,
			device=d_ij.device)
		representation[:, -1] = torch.squeeze(d_ij)
		representation[:, :256] = x[idx_i]
		representation[:, 256:512] = x[idx_j]
		representation[:, 512:512+self.orbitals_size] = atoms_electron_config[idx_i]
		representation[:, 512+self.orbitals_size:512+2 * self.orbitals_size] = atoms_electron_config[idx_j]
		# return the representation
		return representation


class TripleAtomsDistanceAdumbration(torch.nn.Module):
	"""Compute the distance between atoms."""

	def __init__(self,
	             orbitals_size: int = 22,
	             keep_z: bool = False,
              	 include_angles: bool = False):
		"""
		Args:
				cutoff: cutoff radius
		"""
		super(TripleAtomsDistanceAdumbration, self).__init__()
		self.orbitals_size: int = orbitals_size
		self.keep_z: bool = keep_z
		self.angles: bool = include_angles

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
		triplets_pos = self.transform_triplet_coordinates(torch.concat([
			triple_idx_i.view(-1, 1),
			idx_j[triple_idx_j].view(-1, 1),
			idx_j[triple_idx_k].view(-1, 1)
		], dim=1), positions)
		atoms_electron_config = torch.tensor([generate_electron_configurations(i) for i in z.squeeze().tolist()],
		                                     dtype=torch.float32,
		                                     device=idx_j.device)

		# Initializing the zeros tensor to keep the atomic numbers and distances
		triple_representation = torch.zeros(triple_idx_i.size(0), self.orbitals_size * 3 + 3, device=idx_j.device)

		# triple_representation[:, 0:self.orbitals_size] = atoms_electron_config[triple_idx_i]
		# triple_representation[:, self.orbitals_size:2 * self.orbitals_size] = atoms_electron_config[idx_j[triple_idx_j]]
		# triple_representation[:, 2 * self.orbitals_size:3 * self.orbitals_size] = atoms_electron_config[idx_j[triple_idx_k]]
		#
		triple_representation[:, :3 * self.orbitals_size] = torch.cat(
			[
				atoms_electron_config[triple_idx_i],
				atoms_electron_config[idx_j[triple_idx_j]],
				atoms_electron_config[idx_j[triple_idx_k]]
			],
			dim=1)
		# if trip

		# Compute distances
		# r_ij = torch.norm(positions[triple_idx_i] - positions[idx_j[triple_idx_j]], dim=1)
		# r_ik = torch.norm(positions[triple_idx_i] - positions[idx_j[triple_idx_k]], dim=1)
		# r_jk = torch.norm(positions[idx_j[triple_idx_j]] - positions[idx_j[triple_idx_k]], dim=1)

		# Compute angles
		# vec_ij = positions[idx_j[triple_idx_j]] - positions[triple_idx_i]
		# vec_ik = positions[idx_j[triple_idx_k]] - positions[triple_idx_i]
		# vec_jk = positions[idx_j[triple_idx_k]] - positions[idx_j[triple_idx_j]]

		# cos_theta_ijk = torch.sum(vec_ij * vec_ik, dim=1) / (r_ij * r_ik)
		# cos_theta_jki = torch.sum(-vec_ij * vec_jk, dim=1) / (r_ij * r_jk)

		# theta_ijk = torch.acos(cos_theta_ijk)
		# theta_jki = torch.acos(cos_theta_jki)

		# Store distances and angles in the representation
		# triple_representation[:, -3] = r_ij
		# triple_representation[:, -2] = r_ik
		# triple_representation[:, -3] = r_jk
		# triple_representation[:, -1] = cos_theta_ijk
		# triple_representation[:, -1] = theta_jki
		triple_representation[:, -3:] = triplets_pos

		return triple_representation

	def transform_triplet_coordinates(self, triplets, pos):
		"""
		Transform the coordinates of each triplet so that:
		- The first atom (a1) is at (0, 0, 0).
		- The second and third atoms (a2 and a3) have the same x-coordinate,
		  but different y and optionally different z (based on the keep_z flag).

		Args:
			triplets (torch.Tensor): Tensor of triplets (i, j, k) of shape (N, 3).
			pos (torch.Tensor): Positions of atoms (3D coordinates) of shape (M, 3).
			keep_z (bool): If True, preserve the z-coordinate; otherwise, set z to 0.

		Returns:
			torch.Tensor: Transformed coordinates for all triplets of shape (N, 3, 3).
		"""
		# Extract coordinates of the atoms in the triplets
		a1_coords = pos[triplets[:, 0]]  # Shape: (N, 3)
		a2_coords = pos[triplets[:, 1]]  # Shape: (N, 3)
		a3_coords = pos[triplets[:, 2]]  # Shape: (N, 3)

		# Translate a1 to the origin
		a2_relative = a2_coords - a1_coords  # Shape: (N, 3)
		a3_relative = a3_coords - a1_coords  # Shape: (N, 3)

		# Calculate shared x-coordinate
		shared_x = (a2_relative[:, 0] + a3_relative[:, 0]) / 2  # Shape: (N,)

		# Construct transformed coordinates
		if self.keep_z:
			transformed_coords = torch.zeros((triplets.size(0), 5), device=pos.device) # 5 -> 1x, 2y, 2z
			transformed_coords[:, 0] = shared_x  # a2 new position
			transformed_coords[:, 1] = a2_relative[:, 1]  # a2 new position
			transformed_coords[:, 2] = a3_relative[:, 1]  # a2 new position
			transformed_coords[:, 3] = a2_relative[:, 2]  # a2 new position
			transformed_coords[:, 4] = a3_relative[:, 2]  # a2 new position
		else:
			transformed_coords = torch.zeros((triplets.size(0), 3), device=pos.device) # 5 -> 1x, 2y, 2z
			transformed_coords[:, 0] = shared_x  # a2 new position
			transformed_coords[:, 1] = a2_relative[:, 1]  # a2 new position
			transformed_coords[:, 2] = a3_relative[:, 1]  # a2 new position

		return transformed_coords

	def symmetric_coordinates_to_distances(self, triplets, pos):
		"""
		Convert symmetric coordinates of triplets to distances between atoms.
			$f(d_{ij}) + f(d_{ik}) + \oplus (i - j) \cdot (i - k)$

		Args:
			triplets (torch.Tensor): Tensor of triplets (i, j, k) of shape (N, 3).
			pos (torch.Tensor): Positions of atoms (3D coordinates) of shape (M, 3).

		Returns:
			torch.Tensor: Distances between atoms in triplets of shape (N, 3).
		"""
		# Extract coordinates of the atoms in the triplets
		a1_coords = pos[triplets[:, 0]]  # Shape: (N, 3)
		a2_coords = pos[triplets[:, 1]]  # Shape: (N, 3)
		a3_coords = pos[triplets[:, 2]]  # Shape: (N, 3)

		# Calculate distances between atoms
		d_ij = torch.norm(a2_coords - a1_coords, dim=1)  # Shape: (N,)
		d_ik = torch.norm(a3_coords - a1_coords, dim=1)  # Shape: (N,)
		d_jk = torch.norm(a3_coords - a2_coords, dim=1)  # Shape: (N,)
  
		# Calculate angles between atoms
		if self.include_angles:
			# Compute only angle for the center atom
			vec_ij = a2_coords - a1_coords
			vec_ik = a3_coords - a1_coords
			vec_jk = a3_coords - a2_coords
			
			cos_theta_ijk = torch.sum(vec_ij * vec_ik, dim=1) / (d_ij * d_ik)
   

   
			

class MLPDeepSet(nn.Module):

	def __init__(
			self,
			n_atom_basis: int,
			base_cutoff: float,
			inner_cutoff: float,
			outer_cutoff: float,
			max_num_neighbors: int = 32,
			embedding_size: int = 256,
			mlp_layer: int = 5,
			radial_basis: nn.Module = GaussianRBF(20, 15.0),
			use_vector_representation: bool = False,
			forces_based_on_energy: bool = False,
			close_far_split: bool = True,
			using_triplet_module: bool = False,
			rbf_type: str = "gauss",
			trainable_rbf: bool = False,
			dtype: torch.dtype = torch.float32,
	):

		super(MLPDeepSet, self).__init__()

		self.n_atom_basis = n_atom_basis
		self.inner_cutoff = inner_cutoff
		self.outer_cutoff = outer_cutoff # ! This was set self.inner_cutoff by mistake
		self.base_cutoff = base_cutoff
		self.radial_basis = radial_basis
		self.embedding_size = embedding_size
		self.dtype = dtype

		self.use_vector_representation = use_vector_representation
		self.forces_based_on_energy = forces_based_on_energy
		self.close_far_split = close_far_split
		self.using_triplet_module = using_triplet_module
		self.only_oneside = True # ! This is for considering the adjacancy matrix

		self.pair_atoms_coder = PairAtomsDistanceAdumbration()
		self.pair_and_distance_transform = torch.nn.Linear(2 * self.n_atom_basis + 1 + self.radial_basis.n_rbf,
		                                                   self.embedding_size,
		                                                   dtype=dtype)

		self.distance = OptimizedDistance(
		    base_cutoff,
		    outer_cutoff,
		    max_num_pairs=max_num_neighbors,
		    return_vecs=True,
		    loop=True,
		    box=None,
		    long_edge_index=True,
		    check_errors=True, # Set False if there are more than 10k neighbors and it throw an error. Check this thread: https://github.com/torchmd/torchmd-net/issues/203
		)

		self.distance_expansion = rbf_class_mapping[rbf_type](
			base_cutoff, outer_cutoff, self.radial_basis.n_rbf, trainable_rbf
		)
		self.distance_proj = nn.Linear(self.radial_basis.n_rbf, embedding_size, dtype=dtype)
		self.combine = nn.Linear(814, embedding_size, dtype=dtype)
		self.cutoff = CosineCutoff(base_cutoff, outer_cutoff)
		self.embedding = nn.Embedding(100, embedding_size, dtype=dtype)

		self.neighbor_embedding = NeighborEmbedding(embedding_size, 20, base_cutoff, outer_cutoff, 100, dtype)

		self.conv_msgp = tg.nn.SimpleConv()

		try:
			# Initialize MLPs
			in_size: int = embedding_size
			if self.close_far_split:
				self.inner_scalar_mlp = build_mlp(in_size, embedding_size, embedding_size, mlp_layer,
				                                  activation=nn.SiLU)

				self.outer_scalar_mlp = build_mlp(in_size, embedding_size, embedding_size, mlp_layer,
				                                  activation=nn.SiLU)

				if self.use_vector_representation:
					self.inner_vector_mlp = build_mlp(in_size, embedding_size, 3 * embedding_size, mlp_layer,
					                                  activation=nn.SiLU)

					self.outer_vector_mlp = build_mlp(in_size, embedding_size, 3 * embedding_size, mlp_layer,
					                                  activation=nn.SiLU)

			else:
				self.scalar_mlp = build_mlp(in_size, embedding_size, embedding_size, mlp_layer, activation=nn.SiLU)
				if self.use_vector_representation:
					self.vector_mlp = build_mlp(in_size, embedding_size, 3 * embedding_size, mlp_layer,
					                            activation=nn.SiLU)

			if self.using_triplet_module:
				self.triplet_atoms_coder = TripleAtomsDistanceAdumbration()
				self.triplet_scalar_processor_mlp = build_mlp(69, 3 * embedding_size, embedding_size, mlp_layer,
				                                              activation=nn.SiLU)

				if self.use_vector_representation:
					self.triplet_vector_processor_mlp = build_mlp(69, 3 * embedding_size, 3 * embedding_size, mlp_layer,
					                                              activation=nn.SiLU)

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

		self.embedding.reset_parameters()
		self.conv_msgp.reset_parameters()

	def forward(self,
	            z: torch.Tensor,
	            pos: torch.Tensor,
	            batch: torch.Tensor,
	            box: Optional[torch.Tensor] = None,
	            q: Optional[torch.Tensor] = None,
	            s: Optional[torch.Tensor] = None) -> typing.Tuple[
		torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

		# x = torch.tensor([Z_LOGPRIME[atom.item()] for atom in z], device=z.device)

		x = self.embedding(z)

		edge_index, edge_weight, edge_vec = self.distance(pos, batch, box) # TODO: Some sort of sort algorithm have to be applied here to keep the index
		# x_0 = self.conv_msgp(zz, edge_index)


		# print(x_0)
		edge_attr = self.distance_expansion(edge_weight)

		# edge_index = edge_index[:, edge_weight != 0]
		# edge_weight = edge_weight[edge_weight != 0] # * Prevent self loops (No distance between same an atom so it should be 0)
		# edge_vec = edge_vec[edge_vec != 0]
		mask = edge_index[0] != edge_index[1]
		if not mask.all():
			edge_index = edge_index[:, mask]
			edge_weight = edge_weight[mask]
			edge_attr = edge_attr[mask]

		if self.only_oneside:
			edge_index = edge_index[:, ::2]
			edge_weight = edge_weight[::2]
			edge_attr = edge_attr[::2]

		# edge_vec = edge_vec[::2]

		# Compute the cutoff and distance projection
		C = self.cutoff(edge_weight)
		W = self.distance_proj(edge_attr) * C.view(-1, 1)

		# Concatenate W and C
		combined = torch.cat([W, C.view(-1, 1)], dim=1)

		# Get the embeddings of the atomic numbers
		# x_neighbors = self.embedding(z)

		# Combine the original node features with the embeddings



		# sorted_index = torch.argsort(edge_index[0, :])
		# idx_i = edge_index[0, sorted_index]
		# idx_j = edge_index[1, sorted_index]
		# edge_weight = edge_weight[sorted_index]
		# edge_vec = edge_vec[sorted_index]

		idx_i = edge_index[0, :]
		idx_j = edge_index[1, :]
		# phi_ij = self.radial_basis(edge_weight)
		msx = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

		pair_atoms_repr = self.pair_atoms_coder(z, msx, idx_i, idx_j, edge_weight)

		pair_atoms_repr = self.combine(torch.cat([pair_atoms_repr, combined], dim=1))

		q = self.process_pair_scalar(pair_atoms_repr, idx_i)
		mu = self.process_pair_vector(pair_atoms_repr, idx_i) if self.use_vector_representation else None
		# if self.using_triplet_module:
		# 	idx_i_triples, idx_j_triples, idx_k_triples = extra_args["idx_i_triples"], extra_args["idx_j_triples"], \
		# 	extra_args["idx_k_triples"]
		# 	triplet_atoms_repr = self.triplet_atoms_coder(idx_i_triples, idx_j_triples, idx_k_triples, idx_i, idx_j, z,
		# 	                                              pos)
		# 	tq = self.triplet_scalar_pass(triplet_atoms_repr, idx_i, idx_i_triples)
		# 	tmu = self.triplet_vector_pass(triplet_atoms_repr, idx_i,
		# 	                               idx_i_triples) if self.use_vector_representation else None
		#
		# 	return q+tq, mu+tmu, z, pos, batch
		return q, mu, z, pos, batch

	def process_pair_scalar(self, pair_atoms_repr: torch.Tensor, idx_i: torch.Tensor) -> torch.Tensor:
		if self.close_far_split:
			close_data_index = pair_atoms_repr[:, -1] < self.inner_cutoff # ! The tremendous bug was here !!!!!
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
		q = torch.zeros((pair_atoms_repr.size(0), self.embedding_size), device=pair_atoms_repr.device,
		                dtype=pair_atoms_repr.dtype)
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
		mu = torch.zeros((idx_i.size(0), 3, self.embedding_size), device=pair_atoms_repr.device,
		                 dtype=pair_atoms_repr.dtype)
		mu[close_data_index] = self.inner_vector_mlp(pair_atoms_repr[close_data_index]).reshape(
			idx_i[close_data_index].size(0), 3, self.embedding_size)
		mu[far_data_index] = self.outer_vector_mlp(pair_atoms_repr[far_data_index]).reshape(
			idx_i[far_data_index].size(0), 3, self.embedding_size)
		counts = torch.bincount(idx_i)
		buf = torch.zeros((counts.size(0), mu.size(1), mu.size(2)), dtype=mu.dtype, device=mu.device)
		mu = buf.index_add(0, idx_i, mu)
		return mu

	def triplet_scalar_pass(self, triplet_atoms_repr: torch.Tensor, idx_i: torch.Tensor,
	                        triplet_idx_i: torch.Tensor) -> torch.Tensor:
		tq = self.triplet_scalar_processor_mlp(triplet_atoms_repr)
		count = torch.bincount(idx_i)
		buf = torch.zeros((count.size(0), self.embedding_size), dtype=tq.dtype, device=tq.device)
		tq = buf.index_add(0, triplet_idx_i, tq)
		return tq

	def triplet_vector_pass(self, triplet_atoms_repr: torch.Tensor, idx_i: torch.Tensor,
	                        triplet_idx_i: torch.Tensor) -> torch.Tensor:
		tmu = self.triplet_vector_processor_mlp(triplet_atoms_repr).reshape(triplet_atoms_repr.size(0), 3,
		                                                                    self.embedding_size)
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
	# print(z.size())
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
	# print(f"Time taken: {elapsed_time:.6f} seconds")
	# rich.print(edge_index)
	# rich.print(edge_weight)
	# rich.print(edge_vec)
