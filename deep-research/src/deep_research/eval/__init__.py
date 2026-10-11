"""Evaluation harness: programmatic citation verification + LLM-as-judge.

Design rule: anything verifiable by code (does this arXiv id / repo / URL
exist?) is verified by code — reproducible and free. LLM judges only what
code cannot: claim-source alignment and golden key-point coverage.
"""
