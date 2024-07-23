class Config:
    def __init__(self, dataset):
        self.dataset = dataset
        self.processed_dataset = "data/processed_datasets/" + dataset + "/"
        self.tmp_dir = "data/tmp/" + dataset + "/"
        self.localgraph_dir = "data/localgraph/" + dataset + "/"
        self.train_dir = "data/train/" + dataset + "/"
        self.result_dir = "data/results/" + dataset + "/"
        self.MAX_LEN = 32 if dataset == "webqsp" else 64

        self.pretrained_model = {"roberta_base": "model_hub/sup-simcse-roberta-base"}

        self.localgraph = {
            "seed_path": self.tmp_dir + "localgraph_seed.txt",
            "path": self.localgraph_dir + "localgraph.json",
        }

        self.retriever = {
            "train": {
                "input_path": self.train_dir + "train_retriever.csv",
                "output_dir": "model_ckpt/" + dataset + "/retriever/",
            },
            "final_model": "model_ckpt/" + dataset + "/retriever/final_model/",
        }

        self.discriminator = {
            "train": {
                "input_path": self.train_dir + "train_discriminator.csv",
                "output_dir": "model_ckpt/" + dataset + "/discriminator/",
            },
            "final_model": "model_ckpt/"
            + dataset
            + "/discriminator/final_model/discriminator.pt",
        }

        self.inference = {
            "input_path": self.processed_dataset + "test.json",
            "output_dir": self.result_dir,
        }


cfg = Config("webqsp")
