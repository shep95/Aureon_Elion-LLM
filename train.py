#!/usr/bin/env python3
"""
Train and evaluate supervised machine learning models.

This script implements the concepts from the lecture:
  - Supervised machine learning (not mystical "AI")
  - Neural networks as weighting systems
  - Backpropagation to learn weights
  - Facial recognition as the worked example
  - Three required constraints: clean data, measurable goal, defined parameters
"""

from __future__ import annotations

import argparse

from src.face_matcher import (
    demonstrate_edge_case,
    load_face_database,
    train_binary_face_matcher,
    train_face_matcher,
)
from src.neural_network import compare_traditional_vs_ml
from src.synthetic_faces import train_synthetic_face_classifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervised machine learning demo — neural network with backpropagation"
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "identify", "match", "all"],
        default="all",
        help="synthetic: lecture-style features; identify: real faces; match: yes/no pairs",
    )
    parser.add_argument("--people", type=int, default=40, help="Number of people in database")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("SUPERVISED MACHINE LEARNING")
    print("=" * 60)
    print()
    compare_traditional_vs_ml()
    print()
    print("-" * 60)
    print("Three constraints for supervised ML to work:")
    print("  1. Clean data        — labeled face images, not opinions")
    print("  2. Measurable goal   — yes/no match or person ID (not 'what is good?')")
    print("  3. Defined parameters — fixed database of faces with known labels")
    print("-" * 60)
    print()

    if args.mode in ("synthetic", "all"):
        print("=" * 60)
        print("TASK 0: Synthetic face features (eye, nose, chin weights)")
        print("=" * 60)
        print()
        train_synthetic_face_classifier(epochs=args.epochs, seed=args.seed)
        print()

    if args.mode in ("identify", "all"):
        print("=" * 60)
        print("TASK 1: Person identification from real face photos")
        print("=" * 60)
        print()
        network, _ = train_face_matcher(
            num_people=min(args.people, 10),
            epochs=args.epochs,
            seed=args.seed,
        )
        dataset = load_face_database(num_people=min(args.people, 10), seed=args.seed)
        demonstrate_edge_case(network, dataset)
        print()

    if args.mode in ("match", "all"):
        print("=" * 60)
        print("TASK 2: Binary face matching (yes/no)")
        print("=" * 60)
        print()
        train_binary_face_matcher(
            num_people=args.people,
            epochs=args.epochs,
            seed=args.seed,
        )
        print()

    print("=" * 60)
    print("What happened inside the network (the 'black box'):")
    print("  Humans set up the framework: layers, learning rate, labels.")
    print("  The computer discovered the weights via backpropagation.")
    print("  Those weights are not human-readable — that is the black box.")
    print("=" * 60)


if __name__ == "__main__":
    main()
