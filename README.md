# RD-P
This is the official PyTorch implementation for the paper:
> RD-P: A Trustworthy Retrieval-Augmented Prompter with Knowledge Graphs for LLMs (CIKM 2024)

## Knowledge Graph
1. We use Freebase as the knowledge base in our experiment. Please follow [Freebase-Setup](https://github.com/dki-lab/Freebase-Setup) to build a Virtuoso for Freebase.
2. To enhance data access efficiency, we extract the topic-centric subgraphs for all questions in both datasets, converting them to JSON Lines format. The hop count is 2 for WebQSP and 3 for CWQ.

## Dataset
You can download the datasets we use from the links below and place them in the `data/datasets/` folder.
1. WebQuestionsSP (WebQSP): [WebQSP dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52763).
2. Complex Webquestions 1.1 (CWQ): [CWQ dataset](https://allenai.org/data/complexwebquestions).

**Note:** since the original CWQ dataset doesn’t include gold answers for test questions, we use the officially provided SPARQL queries to retrieve answers from Freebase, successfully obtaining answers for 3,269 out of 3,531 questions, forming the test set. To generate the processed test set, run the Jupyter notebook `notebook/get_gold_answers_for_cwq.ipynb`.

## Installation
We implement our approach based on Pytorch and Huggingface Transformers. To install the required Python libraries, please run the following command:

    pip install -r requirements.txt

## Preprocessing
Standardize the data format:

    python preprocess_dataset.py

Use multiprocessing to extract local subgraphs for each topic entity and covert them to JSON Lines format:

    python run_get_localgraph.py

## Training
Generate training data and train the retriever:

    python generate_retriever_data.py

    python train_retriever.py

Generate training data and train the discriminator:

    python generate_discriminator_data.py

    python train_discriminator.py

## Retrieving
Retrieve reasoning paths and candidate answers using both the retriever and discriminator:

    python run_retrieve_path.py
