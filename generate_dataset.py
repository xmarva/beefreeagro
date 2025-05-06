#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
from data_generator import SyntheticDataGenerator

def main():
    parser = argparse.ArgumentParser(description='Synthetic data generator for YOLO training')
    
    parser.add_argument('--stickers_dir', type=str, default='data/raw/stickers',
                        help='Path to directory with stickers')
    parser.add_argument('--backgrounds_dir', type=str, default='data/raw/backgrounds',
                        help='Path to directory with backgrounds')
    parser.add_argument('--output_dir', type=str, default='data/synth/imgs',
                        help='Path to directory for saving generated images')
    parser.add_argument('--annotations_file', type=str, default='annotations.txt',
                        help='File name for saving annotations')
    parser.add_argument('--num_images', type=int, default=1000,
                        help='Number of images to generate')
    parser.add_argument('--overlap_ratio', type=float, default=0.3,
                        help='Maximum stickers overlap ratio')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Seed for random number generator')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.stickers_dir):
        print(f"Error: Stickers directory not found: {args.stickers_dir}")
        return
    
    if not os.path.exists(args.backgrounds_dir):
        print(f"Error: Backgrounds directory not found: {args.backgrounds_dir}")
        return
    
    generator = SyntheticDataGenerator(
        stickers_dir=args.stickers_dir,
        backgrounds_dir=args.backgrounds_dir,
        output_dir=args.output_dir,
        annotations_file=args.annotations_file,
        num_images=args.num_images,
        overlap_ratio=args.overlap_ratio,
        random_seed=args.random_seed
    )
    
    generator.generate_dataset()


if __name__ == "__main__":
    main()