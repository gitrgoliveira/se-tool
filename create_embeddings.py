import argparse
import logging
import os
import sys

from ai.embed_hashicorp import (check_github_token, create_git_embeddings,
                                create_website_embeddings, output_ai,
                                wait_until_finished)

def main(base_path: str, only_missing_embeddings: bool):
    """
    Main function to create embeddings and save them to the specified base path.
    """
    # logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if check_github_token() is None:
        logging.error("No GitHub token found. Exiting.")
        exit(1)
        
    base_path_web = os.path.join(base_path, "web")
    create_website_embeddings(base_path_web, only_missing_embeddings)
    
    base_path_git = os.path.join(base_path, "git")    
    create_git_embeddings(base_path_git, only_missing_embeddings)
        
    # Shutdown the processor
    wait_until_finished()
    logging.info("Finished creating embeddings. Exiting.")


if __name__ == "__main__":
    """
    Parse command line arguments and start the main function.
    """
    parser = argparse.ArgumentParser(description='Create embeddings.')
    parser.add_argument('--base_path', default=output_ai, help='Base path for output')
    parser.add_argument('--only_missing', action='store_true', dest='only_missing_embeddings',
                        help='Only retrieve missing embeddings', default=True)
    args = parser.parse_args()

    main(args.base_path, args.only_missing_embeddings)
