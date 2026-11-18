"""PathSnap: map-matching noisy GPS tracks onto a road network.

Pipeline: load trip -> project to metres -> clean/segment track -> candidate arcs
-> per-candidate features -> classifier probabilities -> HMM/Viterbi decoding
-> stitched route.
"""
