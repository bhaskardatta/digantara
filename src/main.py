"""
Main entry point for GIS Image Analysis Pipeline
Processes TIFF images through detection and segmentation models
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from src.pipeline import GISAnalyzer
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="GIS Image Analysis: AI-powered feature extraction from TIFF imagery"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input TIFF image or directory containing TIFF files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="models/",
        help="Directory containing model weights"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization outputs"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("gis_analyzer", level=log_level)

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Initialize analyzer
        analyzer = GISAnalyzer(
            config=config,
            models_dir=args.models,
            device=args.device,
            batch_size=args.batch_size
        )

        # Process input
        input_path = Path(args.input)
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        if input_path.is_file():
            # Single file processing
            logger.info(f"Processing single file: {input_path}")
            results = analyzer.analyze_image(input_path)
            analyzer.save_results(results, output_path, visualize=args.visualize)
        elif input_path.is_dir():
            # Batch processing
            tiff_files = list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff"))
            logger.info(f"Found {len(tiff_files)} TIFF files for processing")
            
            for tiff_file in tiff_files:
                logger.info(f"Processing: {tiff_file}")
                try:
                    results = analyzer.analyze_image(tiff_file)
                    file_output_dir = output_path / tiff_file.stem
                    analyzer.save_results(results, file_output_dir, visualize=args.visualize)
                except Exception as e:
                    logger.error(f"Failed to process {tiff_file}: {e}")
                    continue
        else:
            logger.error(f"Input path {input_path} is not a valid file or directory")
            sys.exit(1)

        logger.info("Analysis completed successfully!")
        logger.info(f"Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
