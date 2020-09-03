"""
A module for the recursive sampling step of our approach.

It uses an `Axiom` and a `KnowledgeGraph`, and sample entities from the
graph that verify the axiom.
"""


from .sampler import GraphSampler, NaiveGraphSampler, Instances, Sampled