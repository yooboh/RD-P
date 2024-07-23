from discriminator.filter_path import additional_filter
from preprocess.get_answer_names import get_answer_names
from retriever.retrieve_with_discriminator import retrieve_path_with_discriminator

TOP_K = 1
RESERVED_COUNT = 1


def run():
    retrieve_path_with_discriminator(TOP_K)
    additional_filter(TOP_K, RESERVED_COUNT)
    get_answer_names(TOP_K, RESERVED_COUNT)


if __name__ == "__main__":
    run()
