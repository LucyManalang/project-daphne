import datasets
from datasets import load_dataset, Dataset, DatasetDict
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import tempfile
import torch

class Transformer:
    def __init__(self, train_data, seed, valid_data=None, generate=False, prompt=None):
        # from HuggingFace transformers documentation https://huggingface.co/docs/transformers/v4.47.0/en/model_doc/gpt2
        model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2", pad_token="<eos>")

        # from HuggingFace datasets documentation https://huggingface.co/docs/datasets/index
        datasets.utils.logging.set_verbosity_error() # https://github.com/huggingface/datasets/issues/1627 
        train_dataset = Dataset.from_dict({"text": train_data})

        if generate:
            train_dataset = train_dataset.shuffle(seed=seed).select(range(len(train_dataset)))
            dataset = DatasetDict({"train": train_dataset})
        else:
            train_dataset = train_dataset.shuffle(seed=seed).select(range(len(train_dataset) // 500))
            valid_dataset = Dataset.from_dict({"text": valid_data})
            valid_dataset = valid_dataset.shuffle(seed=seed).select(range(len(valid_dataset) // 500))
            dataset = DatasetDict({"train": train_dataset, "valid": valid_dataset})

        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=100,
            num_proc=4
        )

        # from HuggingFace transformers documentation https://huggingface.co/docs/transformers/v4.47.0/en/main_classes/trainer#transformers.TrainingArguments
        training_args = TrainingArguments(
            output_dir=tempfile.mkdtemp(), 
            num_train_epochs=2, 
            per_device_train_batch_size=8,
            logging_dir=None,
            logging_strategy="no",
            log_level="error",
            remove_unused_columns=False,
        )

        self.train(model, training_args, tokenized_dataset)
        
        # generation from https://huggingface.co/docs/transformers/v4.47.0/en/model_doc/gpt2
        if generate:
            device = torch.device("cpu")
            model.to(device)
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

            gen_tokens = model.generate(
                input_ids,
                do_sample=True,
                max_length=len(input_ids[0]) + 256,
                temperature=0.3,
                top_p=0.95,
            )
            self.gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        else:
            self.perplexity = self.calculate_perplexity(model, "".join(dataset["valid"]["text"]))

    # tokenize the dataset, from https://huggingface.co/docs/datasets/v3.1.0/en/package_reference/main_classes#datasets.Dataset.map 
    def tokenize_function(self, examples):
        tokens = self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        tokens["labels"] = tokens["input_ids"].copy()  # Set labels identical to input_ids
        return tokens

    # train the model, from https://huggingface.co/docs/transformers/v4.47.0/en/main_classes/trainer
    def train(self, model, training_args, tokenized_dataset):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            # eval_dataset=tokenized_dataset["valid"]
        )

        trainer.train()
        model.save_pretrained("working_dir")

    # inspired by https://huggingface.co/docs/transformers/v4.47.0/en/perplexity#perplexity-of-fixed-length-models
    def calculate_perplexity(self, model, text):
        model.eval()
        device = torch.device("cpu")
        model.to(device)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        perplexity = torch.exp(loss)
        return perplexity.item()