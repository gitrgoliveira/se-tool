import logging
import os

from ai.embed_hashicorp import (check_github_token, create_git_embeddings,
                                create_website_embeddings, output_ai,
                                wait_until_finished)

if __name__ == "__main__":
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if check_github_token() is None:
        logging.error("No GitHub token found. Exiting.")
        exit(1)
    base_path = os.path.join(output_ai, "web")    
    create_website_embeddings(base_path)
    
    base_path = os.path.join(output_ai, "git")    
    create_git_embeddings(base_path)
    
    # Shutdown the processor
    wait_until_finished()
    logging.info("Finished creating embeddings. Exiting.")
