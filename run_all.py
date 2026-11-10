#### Run the complete Slot Attention experiment. ####

import argparse
from pathlib import Path

from pipeline import run_pipeline


# Load the configuration and execute the complete experiment.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=Path(__file__).with_name("config.json"), type=Path
    )
    arguments = parser.parse_args()

    run_pipeline(arguments.config)


if __name__ == "__main__":
    main()
