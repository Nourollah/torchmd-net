from typing import Callable, Dict, Optional, Union, List
import typing
import rich
from rich import pretty

import torch
import torch.nn as nn
from torch_geometric import nn as g_nn
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


class hp:
	# Training target
	file_path = r"./ethanol.xyz"
	path = r"./train_n.pth"

	# stop file path
	stop_file_path = r"./STOPFILE"

	# Network hyperparameters
	# Save every n steps
	save_steps = 5

	# learning_rate = 0.013
	learning_rate = 0.027

	# Number of steps for secondary gradient descent
	paramsteps = 7
	# Learning rate for secondary gradient descent
	paramlr = 0.001

	# gradient clip value
	clipvalue = 0.01

	# Iterations
	itrtns = 100

	# Batch size - batching is not currently implemented
	batch_size = 20

	max_molecules = 3500

	optimizer = "torch.optim.Adadelta"

	# Number of Lennard-Jones functions
	n_lj = 3

	# Number of Gaussian functions
	n_gauss = 4

	# Number of attention heads
	num_heads = 2

	# Transformer dimension
	dotf = 8

	# Feedforward network dimension
	doff = 6

	# Number of Layers
	nlayers = 2

	# Dropout
	dropout = 0.001

	# Set device to CUDA (GPU) if available (also set in init)
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# Multiplier used to normalise interactions (min should be about 12)
	# Depends upon the threshold
	inm = 12

	# Maximum distance for graph interactions (should be ~2.5-3 for most cases)
	interaction_threshold = 2.5
	# Maximum distance for considering energy interactions
	dist_cut = 3.5

	# lj_gaus constants
	lj_param = 3
	gaus_param = 3

	# number of graph convolution terms
	graphconvterms = 5


# from src import utils
# from src.properties import properties

__all__ = ["Messy"]

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
SWAPPED_ATOMIC_NUMBERS = {v: k for k, v in ATOMIC_NUMBERS.items()}



class MultiHeadAttention(nn.Module):
	def __init__(self):
		super(MultiHeadAttention, self).__init__()
		self.dotf = hp.num_heads * hp.dotf * hp.dotf
		self.num_heads = hp.num_heads
		self.W_q = nn.Linear(self.dotf, self.dotf)
		self.W_k = nn.Linear(self.dotf, self.dotf)
		self.W_v = nn.Linear(self.dotf, self.dotf)
		self.W_o = nn.Linear(self.dotf, self.dotf)
		self.temperature = 1.5
		# Weight initialization
		nn.init.xavier_uniform_(self.W_q.weight)
		nn.init.xavier_uniform_(self.W_k.weight)
		nn.init.xavier_uniform_(self.W_v.weight)
		nn.init.xavier_uniform_(self.W_o.weight)

	def sdp_attention(self, Q, K, V):
		attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (hp.dotf * self.temperature)
		attn_probs = torch.softmax(attn_scores, dim=-1)
		attn_probs = torch.clamp(attn_probs, min=1e-8)
		output = torch.matmul(attn_probs, V)
		return output

	def split_heads(self, x):
		# Handle both batched and unbatched inputs
		if x.dim() == 1:  # Unbatched
			return x.view(1, self.num_heads, hp.dotf * hp.dotf).transpose(1, 2)
		else:  # Batched
			batch_size = x.size(0)
			return x.view(batch_size, self.num_heads, hp.dotf * hp.dotf).transpose(1, 2)

	def combine_heads(self, x):
		# Handle both batched and unbatched inputs
		if x.size(0) == 1:  # Unbatched
			return x.transpose(1, 2).contiguous().view(1, self.dotf)
		else:  # Batched
			batch_size = x.size(0)
			return x.transpose(1, 2).contiguous().view(batch_size, self.dotf)

	def forward(self, Q, K, V):
		# Determine if input is batched
		is_batched = Q.dim() > 1

		# Process input through linear layers
		Q_proj = self.W_q(Q)
		K_proj = self.W_k(K)
		V_proj = self.W_v(V)

		# Split heads
		Q_split = self.split_heads(Q_proj)
		K_split = self.split_heads(K_proj)
		V_split = self.split_heads(V_proj)

		# Apply attention
		attn_output = self.sdp_attention(Q_split, K_split, V_split)

		# Combine heads and project
		output = self.W_o(self.combine_heads(attn_output))

		return output



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.001):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # self.afn = nn.Softplus()  # GELU
        self.afn = nn.GELU()  # GELU

    def forward(self, x):
        return self.fc2(self.dropout(self.afn(self.fc1(x))))


class DecoderLayer(nn.Module):
	def __init__(self):
		super(DecoderLayer, self).__init__()
		self.normx = nn.LayerNorm(hp.num_heads * hp.dotf * hp.dotf)
		self.normy = nn.LayerNorm(hp.num_heads * hp.dotf * hp.dotf)
		self.self_attention = MultiHeadAttention()
		self.norm1 = nn.LayerNorm(hp.num_heads * hp.dotf * hp.dotf)
		self.GCN_dec_attention = MultiHeadAttention()
		self.norm2 = nn.LayerNorm(hp.num_heads * hp.dotf * hp.dotf)
		self.ffn = FeedForward(hp.num_heads * hp.dotf * hp.dotf, hp.doff, hp.dropout)
		self.ffn2 = FeedForward(hp.num_heads * hp.dotf * hp.dotf, hp.doff, hp.dropout)
		self.norm3 = nn.LayerNorm(hp.dotf * hp.dotf * hp.num_heads)
		self.dropout = nn.Dropout(hp.dropout)
		self.afn = nn.GELU()  # GELU
		self.afn1 = nn.GELU()  # GELU

	def forward(self, x, y):
		# Apply layer norms - works with both batched and unbatched inputs
		x_norm = self.normx(x)
		y_norm = self.normy(y)

		# Self-Attention + Add & Norm
		# Our updated MultiHeadAttention handles both batched and unbatched inputs
		self_attn_output = self.dropout(self.self_attention(x_norm, x_norm, x_norm))
		x = x + self_attn_output
		x = self.afn(self.norm1(x + self_attn_output))

		# GCN-Decoder Attention + Add & Norm
		GCN_dec_attn_output = self.GCN_dec_attention(x, x, y_norm)
		x = self.afn1(self.norm2(x + GCN_dec_attn_output))

		# Feed Forward + Add & Norm
		ffn_output = self.ffn(x)
		attn_ffn = self.norm3(x + ffn_output)
		ffn2_output = self.ffn2(attn_ffn)
		x = x + ffn2_output

		return x



class LennardJonesLayer(nn.Module):
    def __init__(self):
        """
        Initialize the Lennard-Jones layer.
        """
        super(LennardJonesLayer, self).__init__()
        # self.initial_epsilon = 0.2
        self.initial_sigma = 3.3
        self.initial_m = 0.001
        self.initial_c = 0.8

    def forward(self, distance, m, c, sigma):
        """
        Forward pass to compute the Lennard-Jones potential.

        Parameters:
        distances (Tensor): Input tensor of distances.
        epsilon (float): Depth of the potential well.
        sigma (float): Distance at which the potential is zero.

        Returns:
        Tensor: Computed Lennard-Jones potentials.
        """
        inv_r = sigma / distance
        inv_r6 = inv_r ** 6
        inv_r12 = inv_r6 * inv_r6

        # Energy
        lj_potential = 4 * c * (inv_r12 - inv_r6)

        # Force (negative gradient of energy)
        lj_force = -4 * c * (((4 * sigma**6)/(distance**7))-((12 * sigma**12)/(distance**13)))
        return lj_potential, lj_force

def lj_func(distance, m, c, sigma):
    lj_potential = 4 * c * (
            ((sigma)/distance)**12
            - ((sigma)/distance)**6)

    return lj_potential


class GaussianLayer(nn.Module):
    def __init__(self):
        """
        Initialize the Gaussian layer.

        """
        super(GaussianLayer, self).__init__()
        # self.amplitude = amplitude
        # self.mean = mean
        # self.stddev = stddev

    def forward(self, distance, amplitude, mean, stddev):
        """
        Forward pass to compute the Gaussian function.

        Parameters:
        distance (Tensor): Input tensor of distances.
        amplitude (float): The amplitude of the Gaussian curve.
        mean (float): The mean (center) of the Gaussian curve.
        stddev (float): The standard deviation (width) of the Gaussian curve.

        Returns:
        Tensor: Computed Gaussian values.
        """
        # Compute the Gaussian function
        exp_term = torch.exp(-((distance - mean) ** 2) / (2 * stddev ** 2))
        # Energy
        gaussian_energy = amplitude * exp_term

        # Force (negative gradient of energy)
        gaussian_force = -(amplitude * (distance - mean) * exp_term * (
            distance - mean)**2 / (stddev**2)
                           )/(stddev**2)
        return gaussian_energy, gaussian_force

def gauss_func(distance, amplitude, mean, stddev):
    gaussian = amplitude * torch.exp(
        -((distance - mean) ** 2) / (2 * stddev ** 2)
    )
    return gaussian


class GausLJLayer(nn.Module):
	def __init__(self):
		super(GausLJLayer, self).__init__()
		self.ljlayers = nn.ModuleList([LennardJonesLayer()
		                               for _ in range(hp.n_lj)])
		self.gauslayers = nn.ModuleList([GaussianLayer()
		                                 for _ in range(hp.n_gauss)])

	def forward(self, distance, lj_gauss_param):
		# For single input handling (scalar or single-element tensor)
		if distance.dim() == 0 or (distance.dim() == 1 and len(distance) == 1):
			return self._forward_single(distance, lj_gauss_param)

		# For batched input
		batch_size = distance.shape[0]
		energies = torch.zeros(batch_size, device=distance.device)
		forces = torch.zeros(batch_size, device=distance.device)

		# Process each batch item
		for i in range(batch_size):
			energy, force = self._forward_single(distance[i], lj_gauss_param[i])
			energies[i] = energy
			forces[i] = force

		return energies, forces

	def _forward_single(self, distance, lj_gauss_param):
		"""Original implementation for a single input"""
		energy = 0
		forces = 0
		index = 0

		for layer in self.ljlayers:
			m = lj_gauss_param[index]
			c = lj_gauss_param[index + 1]
			sigma = lj_gauss_param[index + 2]
			(energya, forcesa) = layer(distance, m, c, sigma)
			forces += forcesa
			energy += energya
			index += hp.lj_param

		for layer in self.gauslayers:
			amplitude = lj_gauss_param[index]
			mean = lj_gauss_param[index + 1]
			stddev = lj_gauss_param[index + 2]
			(energya, forcesa) = layer(distance, amplitude, mean, stddev)
			forces += forcesa
			energy += energya
			index += hp.gaus_param

		return (energy, forces)



def layermapper(distance, lj_gauss_param):
    energy = 0
    index = 0
    for _ in range(hp.n_lj):
        m = lj_gauss_param[index]
        c = lj_gauss_param[index + 1]
        sigma = lj_gauss_param[index + 2]
        energy += lj_func(distance, m, c, sigma)
        index += hp.lj_param
    for _ in range(hp.n_gauss):
        amplitude = lj_gauss_param[index]
        mean = lj_gauss_param[index + 1]
        stddev = lj_gauss_param[index + 2]
        energy += gauss_func(distance, amplitude, mean, stddev)
        index += hp.gaus_param
    return energy


class DecoderModel(nn.Module):
    def __init__(self):
        super(DecoderModel, self).__init__()
        self.embedding = nn.Linear(hp.graphconvterms * 2,
                                   hp.dotf * hp.dotf * hp.num_heads)
        # self.pos_coding = Decoder_layers.Positionalcoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer()
                                     for _ in range(hp.nlayers)])
        self.fc_out = nn.Linear(
                hp.num_heads * hp.dotf * hp.dotf,
                hp.n_lj * hp.lj_param + hp.n_gauss * hp.gaus_param)
        self.ljgauss = GausLJLayer()
        self.dropout = nn.Dropout(hp.dropout)

    def forward(self, distances, gcn_output, is_training=True):
	    if is_training:
		    batch_size = distances.shape[0]

		    # Process all inputs in parallel
		    y = gcn_output.reshape(batch_size, -1)
		    y = self.embedding(y).view(batch_size, hp.num_heads * hp.dotf * hp.dotf)
		    x = y

		    # Apply transformer layers to entire batch
		    for layer in self.layers:
			    x = layer(x, y)

		    # Get parameters for all items
		    lj_gauss_params = self.fc_out(x)

		    # Process all inputs at once with the updated ljgauss
		    energies, forces = self.ljgauss(distances, lj_gauss_params)

		    # Format output to match expected structure (list of tuples)
		    outputs = list(zip(energies, forces))

		    return outputs, lj_gauss_params


class GCNModel(nn.Module):
    def __init__(self):
        super(GCNModel, self).__init__()
        self.conv1 = g_nn.SimpleConv()

    def forward(self, z, edge_index):
        x_1 = self.conv1(z, edge_index)
        x_2 = self.conv1(x_1, edge_index)
        x_3 = self.conv1(x_2, edge_index)
        x_4 = self.conv1(x_3, edge_index)
        x = torch.cat([z, x_1/(hp.inm), x_2/(hp.inm ** 2), x_3/(hp.inm ** 3),
                       x_4/(hp.inm ** 4)], dim=1)

        return x


def normalize_vectors(vectors):
	norms = torch.norm(vectors, dim=-1, keepdim=True)
	norms = torch.where(norms == 0, torch.ones_like(norms), norms)  # Avoid division by zero
	return vectors / norms

# ! Camerons version
# def compute_forces_per_atom(forces, edge_index, vectors, num_atoms):
# 	# Allocate forces_per_atom on hp.device to avoid cross-device operations
# 	forces_per_atom = torch.zeros(num_atoms, 3, device=hp.device)
# 	for atoms, force, vector in zip(edge_index, forces, vectors):
# 		# Extract the force magnitude; assumed to be already on hp.device
# 		force_magnitude = force.to(hp.device)
# 		for dv in vector:
# 			dv = dv.to(hp.device)
# 			force_vector = force_magnitude * dv
# 			forces_per_atom[atoms[0]] += force_vector # ! We deleted this # / 2
# 			forces_per_atom[atoms[1]] -= force_vector # ! We deleted this # / 2
# 	return forces_per_atom


# def compute_energy_per_atom(
# 	energy: list,
# 	edge_index: torch.Tensor,
# 	vectors: torch.Tensor,
# 	num_atoms: torch.Tensor
# ):
#
# 	# Allocate forces_per_atom on hp.device to avoid cross-device operations
# 	energy_per_atom = torch.zeros(num_atoms, 1, device=hp.device)
# 	for atoms, energy, vector in zip(edge_index, energy, vectors):
# 		# Extract the force magnitude; assumed to be already on hp.device
# 		energy_per_atom[atoms[0]] += energy
# 		energy_per_atom[atoms[1]] += energy
# 	return energy_per_atom


def compute_forces_per_atom(
		forces: torch.Tensor,
		edge_index: torch.Tensor,
		vectors: torch.Tensor,
		num_atoms: torch.Tensor,
) -> torch.Tensor:
	# Allocate forces_per_atom on hp.device to avoid cross-device operations
	forces_per_atom = torch.zeros(num_atoms, 3, device=hp.device)

	# Ensure forces has correct shape for broadcasting [N, 1]
	if forces.dim() == 1:
		forces = forces.view(-1, 1)

	# Move vectors to device if needed
	vectors = vectors.to(hp.device)

	# Compute force vectors for all interactions at once
	# Assuming vectors has shape [num_edges, num_dimensions, 3]
	# where num_dimensions is the number of directional vectors per edge

	# Calculate force vectors for all dimensions
	# This handles the inner loop vectorization
	force_vectors = forces.unsqueeze(1) * vectors  # [num_edges, num_dimensions, 3]

	# Sum across dimensions if vectors has multiple dimensions per edge
	if force_vectors.dim() > 2:
		force_vectors = force_vectors.sum(dim=1)  # [num_edges, 3]

	# Use scatter_add_ for accumulating forces efficiently
	# For source atoms (add force)
	source_indices = edge_index[:, 0].view(-1, 1).expand(-1, 3)
	forces_per_atom.scatter_add_(0, source_indices, force_vectors)

	# For destination atoms (subtract force)
	dest_indices = edge_index[:, 1].view(-1, 1).expand(-1, 3)
	forces_per_atom.scatter_add_(0, dest_indices, -force_vectors)  # Note the negation

	return forces_per_atom

def compute_energy_per_atom(
		energy: torch.Tensor,
		edge_index: torch.Tensor,
		vectors: torch.Tensor,
		num_atoms: torch.Tensor
) -> torch.Tensor:
	# Ensure energy has the right shape for broadcasting
	if energy.dim() == 1:
		energy = energy.view(-1, 1)

	# Allocate forces_per_atom on hp.device to avoid cross-device operations
	energy_per_atom = torch.zeros(num_atoms, 1, device=hp.device)
	energy_per_atom.scatter_add_(0, edge_index[:, 0:1], energy)
	energy_per_atom.scatter_add_(0, edge_index[:, 1:1], energy)
	return energy_per_atom


class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.GCN = GCNModel()
		self.decoder = DecoderModel()

	def forward(self, x_coded, edge_index, edge_weight, edge_vec):
		num_atoms = x_coded.size(0)
		# (distance_matrix,
		#  vector_matrix,
		#  graph) = load_data.atoms_to_pyg_data(molecule)
		# Move the graph to the designated device.
		# graph = graph.to(hp.device)
		pooled_atoms = self.GCN(x_coded, edge_index)

		mask = edge_index[:, 0] != edge_index[:, 1]
		filtered_edges = edge_index[mask]

		# Get source and destination indices
		src_indices = filtered_edges[:, 0]
		dst_indices = filtered_edges[:, 1]

		# Get corresponding atoms for both directions
		src_atoms = pooled_atoms[src_indices]
		dst_atoms = pooled_atoms[dst_indices]

		# Create forward and backward bonds in one go
		forward_bonds = torch.cat([src_atoms, dst_atoms], dim=1).unsqueeze(2)
		backward_bonds = torch.cat([dst_atoms, src_atoms], dim=1).unsqueeze(2)
		bonds = torch.cat([forward_bonds, backward_bonds], dim=0)

		# Get corresponding distances and duplicate them for both directions
		edge_distances = edge_weight[src_indices]
		distances = torch.cat([edge_distances, edge_distances], dim=0)

		# Process bonds and distances for the decoder
		bonds = bonds.to(hp.device)
		distances = distances.detach().clone().to(hp.device)
		distances.requires_grad = True

		# Run through decoder
		energiesforces, params = self.decoder(distances, bonds, is_training=True)

		# Convert results into tensors directly without loops
		energies, forces = zip(*energiesforces)

		predicted_energy = torch.stack(energies) if isinstance(energies[0], torch.Tensor) else list(energies)
		predicted_forces = torch.stack(forces) if isinstance(forces[0], torch.Tensor) else list(forces)

		vector_matrix = normalize_vectors(edge_vec)

		predicted_force_per_atom = compute_forces_per_atom(predicted_forces, edge_index, vector_matrix, num_atoms)
		predicted_energy_per_atom = compute_energy_per_atom(predicted_energy, edge_index, vector_matrix, num_atoms)

		return predicted_energy_per_atom, predicted_force_per_atom




class Messy(nn.Module):

	def __init__(
			self,
			base_cutoff: float,
			outer_cutoff: float,
			max_num_neighbors: int = 400,
			embedding_size: int = 256,
			num_rbf=50,
			rbf_type: str = "gauss",
			trainable_rbf: bool = False,
			dtype: torch.dtype = torch.float32,
			skip_duplicates: bool = False,
	):

		super(Messy, self).__init__()

		self.outer_cutoff = outer_cutoff
		self.base_cutoff = base_cutoff
		self.embedding_size = embedding_size
		self.dtype = dtype
		self.skip_duplicates = skip_duplicates


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
			base_cutoff, outer_cutoff, num_rbf, trainable_rbf
		)
		self.distance_proj = nn.Linear(num_rbf, embedding_size, dtype=dtype)
		self.cutoff = CosineCutoff(base_cutoff, outer_cutoff)
		self.embedding = nn.Embedding(100, embedding_size, dtype=dtype)

		self.neighbor_embedding = NeighborEmbedding(embedding_size, 20, base_cutoff, outer_cutoff, 100, dtype)

		self.net = Net()

	def reset_parameters(self):
		...

	def forward(self,
	            z: torch.Tensor,
	            pos: torch.Tensor,
	            batch: torch.Tensor,
	            box: Optional[torch.Tensor] = None,
	            q: Optional[torch.Tensor] = None,
	            s: Optional[torch.Tensor] = None) -> typing.Tuple[
		torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""

		Args:
			z:                                  # Size is like (n_atoms, 1)
			pos:                                # Size is like (n_atoms, 3)
			batch:  						    # Size is like (n_atoms, 1)
			box:            		            # Size is like (3, 3)
			q:                                  # Size is like (n_atoms, 1)
			s:      					        # Size is like (n_atoms, 1)

		Returns:

		"""

		x = self.embedding(z)
		x_coded = torch.tensor([[ELEMENTLOGPRIME[SWAPPED_ATOMIC_NUMBERS[atom.item()]]]
								   for atom in z])

		edge_index, edge_weight, edge_vec = self.distance(pos, batch, box)

		edge_weight: torch.Tensor

		# print(x_0)
		# edge_attr = self.distance_expansion(edge_weight)

		# edge_index = edge_index[:, edge_weight != 0]
		# edge_weight = edge_weight[edge_weight != 0] # * Prevent self loops (No distance between same an atom so it should be 0)
		# edge_vec = edge_vec[edge_vec != 0]
		mask = edge_index[0] != edge_index[1]
		if not mask.all():
			edge_index = edge_index[:, mask]
			edge_weight = edge_weight[mask]
			# edge_attr = edge_attr[mask]
			edge_vec = edge_vec[mask]  # Mask edge_vec as well

		if self.skip_duplicates: # this remove repeated edges in calculation (it means upper triangle matrix)
			edge_index = edge_index[:, ::2]
			edge_weight = edge_weight[::2]
			# edge_attr = edge_attr[::2]
			edge_vec = edge_vec[::2]

		# @TODO: Apply the cosin here
		x_atom_wise, forces = self.net(x_coded, edge_index, edge_weight, edge_vec)


		return x_atom_wise, None, z, pos, batch


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
	model = Messy(
                   base_cutoff=0.0,
                   outer_cutoff=3.0,
                   # radial_basis=None,
                   # use_vector_representation=True,
                   # forces_based_on_energy=False,
                   # close_far_split=True,
                   # using_triplet_module=True
	)

	# Generate random input
	z = torch.randint(1, 100, (18, 1))
	# print(z.size())
	pos = torch.randint(0, 5, (18, 3), dtype=torch.float32)
	batch = torch.zeros(100, dtype=torch.long)
	# box = torch.rand(3, 3)
	x, vec, z, pos, batch = model(z, pos, batch)
