import numpy as np

import os
import json
import argparse
from functools import partial
from typing import Dict, List, NamedTuple, Optional, Union, Any

import torch.nn as nn

import optuna
from optuna.pruners import NopPruner
from transformers import TrainingArguments, EvalPrediction, TextClassificationPipeline, BertTokenizerFast, AutoTokenizer, EncoderDecoderModel, BertTokenizer, DataCollatorForSeq2Seq

import adapters
from evaluate import load
from datasets import Dataset, load_dataset, load_metric
from adapters import BertAdapterModel, AdapterTrainer, AutoAdapterModel
from adapters import ConfigUnion, ParBnConfig, PrefixTuningConfig, DoubleSeqBnConfig, PrefixTuningConfig, LoRAConfig

def load_datetset(dataset_name, seed = 2024):
    """Loads and splits the dataset into train, val and test"""
    # Load Data
    if dataset_name == "wmt16":
        dataset = load_dataset("wmt16", "ro-en")
    elif dataset_name in ["rte", "boolq", "copa"]:
        dataset = load_dataset("super_glue", dataset_name)
    else:
        dataset = load_dataset(dataset_name)
    # Train-Validation Split. Subsetting data if needed
    if dataset_name == "nyu-mll/multi_nli":
        # MNLI dataset has matched and mismatched validation sets;
        # MAM paper used matched as validation and mismatched as test set;
        dataset["train"] = dataset["train"]
        dataset["validation"] = load_dataset(dataset_name, split="validation_matched")
        dataset["test"] = load_dataset(dataset_name, split="validation_mismatched")
        del dataset["validation_matched"]
        del dataset["validation_mismatched"]
    elif dataset_name == "stanfordnlp/imdb":
        split = dataset["train"].train_test_split(.2, seed=seed)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]
        del dataset["unsupervised"]
    elif dataset_name == "EdinburghNLP/xsum":
        # Randomly sample a subset from the validation set to prevent RAM explode
        dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(range(500)) # otherwise memory explodes, depend on model size
    elif dataset_name == "wmt16":
        # Randomly sample a subset from the validation set to prevent RAM explode
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(int(len(dataset["train"]) * 0.3))) # to save time, otherwise >1 hour
        dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(range(10)) # otherwise memory explodes
    # Not needed for now - occupies memory
    del dataset["test"]
    return dataset

def tokenize_batch(batch, tokenizer, dataset_name):
    """Encodes a batch of input data using the model tokenizer."""
    # Tokenize the data
    if dataset_name == "stanfordnlp/sst2": # Paper used 512 - but 16GB GPU would explode
        return tokenizer(batch["sentence"], max_length=256, truncation=True, padding="max_length")
    elif dataset_name == "stanfordnlp/imdb":
        return tokenizer(batch["text"], max_length=256, truncation=True, padding="max_length")
    elif dataset_name in ["nyu-mll/multi_nli", "rte"]: # Paper used 512 - but 16GB GPU would explode
        inputs = []
        for premise, hypothesis in zip(batch["premise"], batch["hypothesis"]):
            inputs.append(f"{premise} [SEP] {hypothesis}")
        return tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    elif dataset_name == "boolq":
        inputs = []
        for passage, question in zip(batch["question"], batch["passage"]):
            inputs.append(f"{passage} [SEP] {question}")
        return tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    elif dataset_name == "copa":
        inputs = []
        for premise, question, choice1, choice2 in zip(batch["premise"], batch["question"], batch["choice1"], batch["choice2"]):
            inputs.append(f"{premise} {question} {choice1}")
            inputs.append(f"{premise} {question} {choice2}")
        tokenized_examples = tokenizer(inputs, max_length=256, truncation=True, padding=True)
        return {k: [v[i : i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}
    elif dataset_name == "EdinburghNLP/xsum": # Paper used 512 - but 16GB GPU would explode
        inputs = ["summarize: " + doc for doc in batch["document"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True) #padding="max_length"
        labels = tokenizer(text_target=batch["summary"], max_length=64, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    elif dataset_name == "wmt16": # Paper used 150
        inputs = ["translate English to Romanian: " + example["en"] for example in batch["translation"]]
        targets = [example["ro"] for example in batch["translation"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=64, truncation=True)
        return model_inputs
    else: raise ValueError(f"Unsupported dataset: {dataset_name}")

def remove_unused_cols(dataset):
    """Removes unused columns from the dataset that are not required for training or evaluation."""
    keep_cols = ["input_ids", "attention_mask", "labels", 'token_type_ids']
    remove_cols = [col for col in dataset["train"].column_names if col not in keep_cols]
    dataset["train"] = dataset["train"].remove_columns(remove_cols)
    dataset["validation"] = dataset["validation"].remove_columns(remove_cols)
    if "test" in dataset: dataset["test"] = dataset["test"].remove_columns(remove_cols)
    return dataset

def preprocess_dataset(dataset_name, tokenizer):
    """Preporcess datasets into transformer-compatible input format"""
    # Load data
    dataset = load_datetset(dataset_name)
    # Tokenize the input data
    tokenize_batch_partial = partial(tokenize_batch, tokenizer = tokenizer, dataset_name = dataset_name)
    dataset = dataset.map(tokenize_batch_partial, batched=True)
    # Prepare dataset format based on tasks
    if dataset_name in ["EdinburghNLP/xsum", "wmt16"]:
        # Transform to pytorch tensors and only output the required columns
        dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    elif dataset_name in ["stanfordnlp/sst2", "nyu-mll/multi_nli", "rte", "stanfordnlp/imdb", "boolq", "copa"]:
        # The transformers model expects the target class column to be named "labels"
        dataset = dataset.rename_column("label", "labels")
        # Transform to pytorch tensors and only output the required columns
        dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    # Remove unused cols in the dataset
    dataset = remove_unused_cols(dataset)
    return dataset

def count_parameters(model, adapter_name, adapter_setting):
    """Counts and prints the number of trainable parameters in the model, including adapter-specific parameters."""
    prefix_tuning_params = 0
    par_bn_params = 0
    trainable_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            if adapter_name in name:
                if "prefix_tuning" in name:
                    prefix_tuning_params += param.numel()
                if "adapters" in name:
                    par_bn_params += param.numel()
        total_params += param.numel()
    adapter_params = prefix_tuning_params + par_bn_params
    if AUTO_FLAG: 
        target_alloc_ratio = adapter_setting["budget_alloc"]
    else:
        adapter_setting["param_budget"] = round(adapter_params / (total_params-adapter_params), 5)
        adapter_setting["budget_alloc"] = round(prefix_tuning_params / adapter_params, 2)

    # Print Results
    print(f"\nAdapter Parameters:")
    if "mam" in adapter_name or prefix_tuning_params: print(f"\tPrefixTuningConfig: {prefix_tuning_params} trainable parameters")
    if "mam" in adapter_name or par_bn_params: print(f"\tParBnConfig: {par_bn_params} trainable parameters")
    if "mam" in adapter_name: 
        if AUTO_FLAG: print(f"\tActual allocation ratio: {prefix_tuning_params/adapter_params:.2f} against target ratio {target_alloc_ratio}")
        else: print(f"\tCurrent allocation ratio: {prefix_tuning_params/adapter_params:.2f}")
    print(f"\nAdapter Parameters Ratio: {adapter_params}/{total_params-adapter_params} = {adapter_params / (total_params-adapter_params):.2%}\n")
    print(f"Total trainable Parameters Ratio: {trainable_params}/{total_params-adapter_params} = {trainable_params / (total_params-adapter_params):.2%}\n")

def get_adapter_name(adapter_setting):
    """Generates a name for the adapter based on the adapter settings."""
    if AUTO_FLAG:
        return "{adapter}_budget_{budget}_alloc_{budget_alloc}_scaling_{scaling}".format(
            adapter=adapter_setting["adapter"],
            budget=str(adapter_setting["param_budget"]).replace(".","-"),
            budget_alloc=str(adapter_setting["budget_alloc"]).replace(".","-"),
            scaling=str(adapter_setting["scaling"]).replace(".","-")
        )
    
    else:
        return "{adapter}_-l_{bn_size}_-r_{reduction_factor}_-lr_{bn_size_for_par}_-s_{scaling}".format(
            adapter=adapter_setting["adapter"],
            bn_size=str(int(adapter_setting["bottleneck_size"])) if adapter_setting["bottleneck_size"] else "None",
            reduction_factor=str(adapter_setting["reduction_factor"]).replace(".","-"),
            bn_size_for_par=str(int(adapter_setting["par_bottleneck_size"])) if adapter_setting["par_bottleneck_size"] else "None",
            scaling=str(adapter_setting["scaling"]).replace(".","-")
        )

def conversion_between_reduction_factor_and_bottleneck(model, r = None, pl = None, scaling = 4):
    """Converts between the reduction factor and bottleneck size for the Adapter layer"""
    # Given reduction factor -r, convert it to corresponding bottleneck size -pl for Parallel Adapter
    if r:
        adapter_config = ParBnConfig(reduction_factor=r, scaling=scaling)
        model.add_adapter("par_test", config=adapter_config, set_active=False)
        model.train_adapter("par_test")
        for name, module in model.named_modules():
            if "par_test" in name and hasattr(module, 'adapter_down'):
                adapter_down = module.adapter_down
                if isinstance(adapter_down, nn.Sequential):
                    for layer in adapter_down:
                        if isinstance(layer, nn.Linear):
                            model.set_active_adapters(None)
                            model.delete_adapter("par_test")
                            return int(layer.out_features)
    
    # Given bottleneck size -pl, convert it to corresponding reduction factor -r for Parallel Adapter
    elif pl:
        adapter_config = ParBnConfig(reduction_factor=1.0, scaling=scaling)
        model.add_adapter("par_test", config=adapter_config, set_active=False)
        model.train_adapter("par_test")
        for name, module in model.named_modules():
            if "par_test" in name and hasattr(module, 'adapter_down'):
                adapter_down = module.adapter_down
                if isinstance(adapter_down, nn.Sequential):
                    for layer in adapter_down:
                        if isinstance(layer, nn.Linear):
                            in_feat = int(layer.in_features)
                            model.set_active_adapters(None)
                            model.delete_adapter("par_test")
                            return float(in_feat / pl)


def binary_search_adapter_config(model, adapter_option, adapter_budgets, scaling, head_param):
    """Performs binary search to find the optimal adapter configuration based on the parameter budget."""
    global adapter_setting
    if adapter_option == "prefix":
        # Binary search for Prefix Tuning bottleneck size
        l_left, l_right = 1, 2000
        while l_left <= l_right:
            l_mid = (l_left + l_right) // 2
            adapter_config = PrefixTuningConfig(bottleneck_size=l_mid)
            model.add_adapter("prefix", config=adapter_config, set_active=False)
            model.train_adapter("prefix")
            adapter_param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad) - head_param
            if adapter_param_cnt >= adapter_budgets:
                l_right = l_mid - 1
            else:
                l_left = l_mid + 1
            model.set_active_adapters(None)
            model.delete_adapter("prefix")
        adapter_setting["bottleneck_size"] = l_left
        return PrefixTuningConfig(bottleneck_size=l_left)

    if adapter_option == "par":
        # Binary search for Parallel Adapter reduction factor
        r_left, r_right = 1e-5, 100.0 
        delta = 1e-5
        while r_left <= r_right:
            r_mid = (r_left + r_right) / 2
            adapter_config = ParBnConfig(reduction_factor=r_mid, scaling=scaling)
            model.add_adapter("par", config=adapter_config, set_active=False)
            model.train_adapter("par")
            adapter_param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad) - head_param
            if adapter_param_cnt <= adapter_budgets:
                r_right = r_mid - delta
            else:
                r_left = r_mid + delta
            model.set_active_adapters(None)
            model.delete_adapter("par")
        adapter_setting["reduction_factor"] = r_left
        adapter_setting["par_bottleneck_size"] = conversion_between_reduction_factor_and_bottleneck(model, r=r_left, scaling=scaling)
        return ParBnConfig(reduction_factor=r_left, scaling=scaling)

def init_adapter(model, adapter_budgets, head_param):
    """Initializes the adapter configuration based on the adapter settings and parameter budget."""
    # Prefix-Tuning Adapter
    if adapter_setting["adapter"] == "prefix":
        # If AUTO mode, do configuration search below
        if AUTO_FLAG: return binary_search_adapter_config(model, adapter_setting["adapter"], adapter_budgets, None, head_param)
        # If MANUAL mode, return configuration directly
        else: return PrefixTuningConfig(bottleneck_size=adapter_setting["bottleneck_size"])

    # Parallel Adapter by He
    elif adapter_setting["adapter"] == "par":
        # If AUTO mode, do configuration search below
        if AUTO_FLAG: return binary_search_adapter_config(model, adapter_setting["adapter"], adapter_budgets, adapter_setting["scaling"], head_param)
        # If MANUAL mode, return configuration directly
        else: return ParBnConfig(reduction_factor=adapter_setting["reduction_factor"], scaling = adapter_setting["scaling"])

    # Mix-And-Match Adapter by He
    elif adapter_setting["adapter"] == "mam":
        # If AUTO mode, do configuration search below
        if AUTO_FLAG:
            prefix_budget = adapter_budgets * adapter_setting["budget_alloc"]
            par_budget = adapter_budgets * (1-adapter_setting["budget_alloc"])
            if prefix_budget == 0:
                return binary_search_adapter_config(model, "par", par_budget, adapter_setting["scaling"], head_param)
            elif par_budget == 0:
                return binary_search_adapter_config(model, "prefix", prefix_budget, None, head_param)
            else:
                return ConfigUnion(
                        binary_search_adapter_config(model, "prefix", prefix_budget, None, head_param),
                        binary_search_adapter_config(model, "par", par_budget, adapter_setting["scaling"], head_param),
                    )
        # If MANUAL mode, return configuration directly
        else: return ConfigUnion(
                    PrefixTuningConfig(bottleneck_size=int(adapter_setting["bottleneck_size"])),
                    ParBnConfig(reduction_factor=adapter_setting["reduction_factor"], scaling=adapter_setting["scaling"]),
                )
    
    else: raise ValueError(f"Unsupported adapter setting: {adapter_setting}")

def inject_adapter_and_head(model, dataset_name):
    """Injects the adapter and output head into the model based on the dataset and adapter settings."""
    global adapter_setting

    # Compute parameter counts - pretrained model
    pretrained_param = sum(p.numel() for p in model.parameters())

    # Add a matching output head depending on task types
    if dataset_name in ["stanfordnlp/sst2", "stanfordnlp/imdb", "rte", "boolq"]: 
        model.delete_head("binary_classification")
        model.add_classification_head("binary_classification", num_labels=2)
    elif dataset_name in ["nyu-mll/multi_nli"]: 
        model.delete_head("triplet_classification")
        model.add_classification_head("triplet_classification", num_labels=3)
    elif dataset_name in ["copa"]:
        model.delete_head("multi_choice")
        model.add_multiple_choice_head("multi_choice", num_choices=2)
    elif dataset_name in ["EdinburghNLP/xsum", "wmt16"]: pass
    else: raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Compute parameter counts - total and head
    total_param = sum(p.numel() for p in model.parameters())
    head_param = total_param - pretrained_param

    # Compute parameter budgets - adapter
    adapter_budgets = int(total_param * adapter_setting["param_budget"]) if AUTO_FLAG else None

    # Initiate Adapters based on the budgets
    global adapter_name, target_adapter_config
    # Prevent adapater being initialized twice in init_trainer() and hyperparameter_search()
    if target_adapter_config is None:
        # If specified bottleneck size for parallel adapter module, convert it to reduction factor. Vice versa.
        if "par_bottleneck_size" in adapter_setting:
            adapter_setting["reduction_factor"] = conversion_between_reduction_factor_and_bottleneck(model, pl=adapter_setting["par_bottleneck_size"], scaling=adapter_setting["scaling"])
        elif "reduction_factor" in adapter_setting:
            adapter_setting["par_bottleneck_size"] = conversion_between_reduction_factor_and_bottleneck(model, r=adapter_setting["reduction_factor"], scaling=adapter_setting["scaling"])
        target_adapter_config = init_adapter(model, adapter_budgets, head_param)
        print("\nAdapter Configuration:\n{}".format(target_adapter_config.__dict__))
        if "reduction_factor" in adapter_setting and "par_bottleneck_size" in adapter_setting:
            print("\nEquivalent reduction factor and bottleneck size for Parallel Adapter: r={} <---> pl={}\n".format(adapter_setting["reduction_factor"], adapter_setting["par_bottleneck_size"]))

    # Deactivate all adapters
    model.set_active_adapters(None)
    model.delete_adapter("prefix")
    model.delete_adapter("par")
    model.delete_adapter("par_test")

    # Insert adapters of interest into the model
    adapter_name = get_adapter_name(adapter_setting)
    model.add_adapter(adapter_name, config=target_adapter_config, set_active=False)

    # Activate the adapter (only the ones in the list):
    # 1. It freezes all weights of the pre-trained model, so only the adapter weights are updated during training.
    # 2. It activates the adapter and the prediction/classification head such that both are used in every forward pass.
    model.train_adapter(adapter_name)
    print()

    # Show summaries of trainable parameters and adapter parameters - only print once
    global print_params
    if print_params: count_parameters(model, adapter_name, adapter_setting); print_params = False

    return model

def get_metrics(dataset_name, tokenizer):
    """"Returns the appropriate evaluation metric based on the dataset."""
    # Accuracy metric - SST-2, MNLI, RTE, COPA, BoolQ, and IMDB
    def accuracy(p: EvalPrediction):
        # Get predictions
        preds, labels = p
        preds = np.argmax(preds, axis=-1)

        accuracy = load("accuracy")
        result = accuracy.compute(predictions=preds, references=labels)
        result = {k: round(v, 4) for k, v in result.items() if k not in []}
        print("\nEval results for this epoch:", result)

        global eval_metrics_recorder
        eval_metrics_recorder.append(result)
    
        return result
    
    # ROUGE metric - XSum
    def rouge(p: EvalPrediction):
        # Get predictions
        preds, labels = p

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.argmax(preds, axis=-1)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Get Labels: Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute rouge scores
        rouge = load("rouge")
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v, 4) for k, v in result.items() if k in ["rouge1","rouge2","rougeL"]}
        print("\nEval results for this epoch:", result)
        
        global eval_metrics_recorder
        eval_metrics_recorder.append(result)
    
        return result
    
    # BLEU metric - WMT16
    def bleu(p: EvalPrediction):
        
        preds, labels = p

        if isinstance(preds, tuple):
            preds = preds[0]
        
        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [[label.strip()] for label in labels]
            return preds, labels
        
        preds = np.argmax(preds, axis=-1)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        bleu = load("sacrebleu")
        result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        result = {k: round(v, 4) for k, v in result.items() if k in ["bleu"]}
        print("\nEval results for this epoch:", result)

        global eval_metrics_recorder
        eval_metrics_recorder.append(result)
        
        return result
    
    # Assign metrics based on tasks
    if dataset_name in ["stanfordnlp/sst2", "nyu-mll/multi_nli", "rte", "stanfordnlp/imdb", "boolq", "copa"]:
        return accuracy
    elif dataset_name == "EdinburghNLP/xsum":
        return rouge
    elif dataset_name == "wmt16":
        return bleu
    else: raise ValueError(f"Unsupported dataset: {dataset_name}")

def init_model(trial: Any, model_name: str, dataset_name:str, drop_out_ratio):
    """Initializes the model based on the dataset, adapter settings, and dropout ratio."""
    if dataset_name in ["stanfordnlp/sst2", "rte", "stanfordnlp/imdb", "boolq", "copa"]: 
        model_init = AutoAdapterModel.from_pretrained(model_name, num_labels=2, hidden_dropout_prob=drop_out_ratio, attention_probs_dropout_prob=drop_out_ratio)
    elif dataset_name == "nyu-mll/multi_nli":
        model_init = AutoAdapterModel.from_pretrained(model_name, num_labels=3, hidden_dropout_prob=drop_out_ratio, attention_probs_dropout_prob=drop_out_ratio)
    elif dataset_name in ["EdinburghNLP/xsum", "wmt16"]:
        model_init = AutoAdapterModel.from_pretrained(model_name)
        adapters.init(model_init)
    else: raise ValueError(f"Unsupported dataset: {dataset_name}")

    if dataset_name not in ["EdinburghNLP/xsum", "wmt16"]: model_init.delete_head("default") # Wired default heads added in bert-base-uncase, removed here
    model_init = inject_adapter_and_head(model_init, dataset_name)

    return model_init

def init_trainer(model_name: str, dataset_name, metric, drop_out_ratio, save_total_limit = 5):
    """Initializes the AdapterTrainer with the specified model, dataset, metric, and adapter settings."""
    # Note that adapter training usually requires a few more training epochs than full fine-tuning.
    training_args = TrainingArguments(
        logging_strategy="no",
        evaluation_strategy="epoch",
        output_dir="./training_outputs/"+dataset_name.replace("/","-"),
        overwrite_output_dir=True,
        save_total_limit=save_total_limit,
        save_strategy = "epoch",
        remove_unused_columns=False, # ensure the dataset labels are properly passed to the model
        load_best_model_at_end=True,
        #save_only_model=True,
    )

    # Mostly similar to Trainer() except a few tweaks for Adapters, e.g. checkpointing only adapter weights
    if dataset_name in ["EdinburghNLP/xsum", "wmt16"]:
        trainer = AdapterTrainer(
            model = None,
            model_init= partial(init_model, model_name=model_name, dataset_name=dataset_name, drop_out_ratio = drop_out_ratio),
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            compute_metrics=metric,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name),
        )

    else:
        trainer = AdapterTrainer(
            model = None,
            model_init= partial(init_model, model_name=model_name, dataset_name=dataset_name, drop_out_ratio = drop_out_ratio),
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            compute_metrics=metric,
        )

    return trainer

def hyperparameter_search_settings(search_space) -> Dict[str, Any]:
    """Returns keyword arguments passed to Trainer.hyperparameter_search."""

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", search_space["learning_rate"]),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", search_space["per_device_train_batch_size"]),
            "num_train_epochs": trial.suggest_categorical("num_train_epochs", search_space["num_train_epochs"]),
            "warmup_ratio": trial.suggest_categorical("warmup_ratio", search_space["warmup_ratio"]),
            "label_smoothing_factor": trial.suggest_categorical("label_smoothing_factor", search_space["label_smoothing_factor"]),
            "weight_decay": trial.suggest_categorical("weight_decay", search_space["weight_decay"]),
            "max_grad_norm": trial.suggest_categorical("max_grad_norm", search_space["max_grad_norm"]),
        }
    
    return {
        "direction": "maximize",
        "backend": "optuna",
        "hp_space": optuna_hp_space,
        "sampler": optuna.samplers.GridSampler(search_space),
        "n_trials": 500,
        #"pruner": optuna.pruners.NopPruner()
    }

if __name__ == "__main__":
    # Define command-line arguments. 
    # AUTO-MODE example 1: python model.py -T sst2 -A par -P 0.01 -s 4 -bs 32 64 -lr 1e-4 5e-5 -ep 3 5 8 -wr 0.06
    # AUTO-MODE example 2: python model.py -T sst2 -A mam -P 0.01 -B 0.5 -s 4 -bs 32 64 -lr 1e-4 5e-5 -ep 3 5
    # AUTO-MODE example 3: python model.py -T sst2 -A prefix -P 0.01 -s 4 -bs 32 -lr 1e-4 -ep 2
    # MANUAL-MODE example 4: python model.py -T sst2 -A prefix -l 100 -s 4 -bs 32 -lr 1e-4 -ep 2
    # MANUAL-MODE example 5: python model.py -T sst2 -A mam -l 100 -r 1.5 -bs 32 -lr 1e-4 -ep 2
    parser = argparse.ArgumentParser()

    # Main argumemts: Task, Adapter, Total Parameter Budget, Paramter Allocation(not required if not using MAM)
    # Auto Mode: If -P specified, the program will automatically identify the best-fitted adapter configurations -l and -r - no need to manually specify -l and -r
    # Note: -s does not affect parameter count and still needed to be configured for MAM and Parallel Adapter
    # Note: Accept 1 input only for each argument, as running 1 combo is already very time-costly. If want to try multiple in one shot, using bash script (HW3) is recommended.
    parser.add_argument("-T", "--task", type=str, default=None, help="Task type for tuning: sst2, mnli, xsum, wmt16, rte, imdb, boolq, copa")
    parser.add_argument("-A", "--adapter", type=str, default="mam", help="Choice of adapters: par (parallel adapter), prefix, mam")
    parser.add_argument("-P", "--param-budget", type=float, default=None, help="Parameter budget for the adapter = % of pretrained model parameters")
    parser.add_argument("-B", "--budget-allocation", type=float, default=None, help="Percentage of parameter budgets allocated to attention block (prefix), like 0.5")

    # Adapter arguments
    # Manual Mode: If -P not specified, -l and -s will be used to manually configure the adapter - use this mode when ad-hoc testing is needed 
    # Note: Accept 1 input for each argument -  use this mode when ad-hoc testing is needed
    parser.add_argument("-s", "--scaling-factor", type=float, default=1, help="Scaling factor for Parallel Adapter - the paper used 4 in experiments.")
    parser.add_argument("-l", "--bottleneck-size", type=float, default=None, help="bottleneck_size for PrefixTuning - LARGER -> bigger size")
    parser.add_argument("-r", "--reduction-factor", type=float, default=None, help="Scaling factor for Parallel Adapter - LARGER -> smaller size. when > 1.0, it does downward projection")
    parser.add_argument("-pl", "--par-bottleneck-size", type=float, default=None, help="bottleneck_size for Parallel Adapter = layer_input_size / reduction_factor")

    # Model arguments: Parameter sets for model hyperparameter tuning
    # Note: Accept > 1 inputs for each arguments, as many as needed to search for the best model setting for the adapter you choose
    parser.add_argument("-bs", "--batch-size", type=int, nargs="+", default=[32], help="A list of batch sizes for hyperparameter tuning")
    parser.add_argument("-lr", "--learning-rate", type=float, nargs="+", default=None, help="A list of learning rates for hyperparameter tuning")
    parser.add_argument("-ep", "--num-epochs", type=int, nargs="+", default=[10], help="A list of num_epochs for hyperparameter tuning. Paper used 10 for SST-2 task.")
    parser.add_argument("-wr", "--warmup-ratio", type=float, nargs="+", default=None, help="In paper, 0 for xsum and wmt16, 0.06 for others")
    parser.add_argument("-ls", "--label-smoothing-factor", type=float, nargs="+", default=None, help="In paper, 0.1 for xsum and wmt16, 0 for others")
    parser.add_argument("-wd", "--weight-decay", type=float, nargs="+", default=None, help="In paper, 0.01 for xsum, 0.1 for others")
    parser.add_argument("-mg", "--max-grad-norm", type=float, nargs="+", default=None, help="In paper, 0.1 for xsum, 1 for others")
    args = parser.parse_args()

    # Model name: bert-base-uncased, prajjwal1/bert-medium, prajjwal1/bert-tiny, albert/albert-base-v2
    # google-t5/t5-small, google/mt5-small, google/t5-efficient-mini
    if args.task in ["sst2", "mnli", "rte", "imdb", "copa", "boolq"]:
        model_name = 'bert-base-uncased' #bert-base-uncased
    elif args.task == "xsum":
        model_name = "google/t5-efficient-mini" # google-t5/t5-small
    elif args.task == "wmt16":
        model_name = 'google/mt5-small' #google/mt5-base

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Load dataset and encode data: stanfordnlp/sst2, nyu-mll/multi_nli, EdinburghNLP/xsum, wmt16/ro-en, rte, stanfordnlp/imdb, copa, boolq
    dataset_dict = {"sst2":"stanfordnlp/sst2", "mnli":"nyu-mll/multi_nli", "xsum":"EdinburghNLP/xsum", "wmt16":"wmt16", "rte":"rte", "copa":"copa", "boolq":"boolq", "imdb":"stanfordnlp/imdb"}
    assert args.task in dataset_dict, f"Invalid task: {args.task}. Available tasks: {list(dataset_dict.keys())}"
    dataset_name = dataset_dict[args.task]
    dataset = preprocess_dataset(dataset_name, tokenizer)

    # Decide adpter to use - Use AUTO or MANUAL model according to command-line args
    global AUTO_FLAG, target_adapter_config, print_params
    target_adapter_config = None
    print_params = True
    AUTO_FLAG = 1 if args.param_budget else 0
    assert args.adapter in ["par","prefix","mam"], "Invalid task: {}. Available tasks: {}".format(args.adapter, ["par","prefix","mam"])
    if AUTO_FLAG == 0 and args.adapter == "par": assert args.reduction_factor or args.par_bottleneck_size, "-r or -rl missing - required if you choose parallel adapter in manual mode"
    if AUTO_FLAG == 0 and args.adapter == "prefix": assert args.bottleneck_size, "-l missing - required if you choose prefix tuning in manual mode"
    if AUTO_FLAG == 0 and args.adapter == "mam": assert args.bottleneck_size and (args.reduction_factor or args.par_bottleneck_size), "-l or (-r / -rl) missing - required if you choose MAM in manual mode"
    if AUTO_FLAG == 0 and args.adapter != "prefix": assert args.reduction_factor is None or args.par_bottleneck_size is None, "Either -r or -rl should be specified. You can't specifiy both at the same time"

    # Set adapter setting
    global adapter_setting
    if AUTO_FLAG == 1:
        adapter_setting = {"adapter":args.adapter, "param_budget":args.param_budget, "budget_alloc": args.budget_allocation, "scaling":args.scaling_factor}
    elif AUTO_FLAG == 0 and args.reduction_factor: 
        adapter_setting = {"adapter":args.adapter, "reduction_factor":args.reduction_factor, "bottleneck_size":args.bottleneck_size, "scaling":args.scaling_factor}
    elif AUTO_FLAG == 0 and args.par_bottleneck_size:
        adapter_setting = {"adapter":args.adapter, "par_bottleneck_size":args.par_bottleneck_size, "bottleneck_size":args.bottleneck_size, "scaling":args.scaling_factor}

    # Set Training Arguments and Initiate Trainer
    metric = get_metrics(dataset_name, tokenizer)
    trainer = init_trainer(model_name, dataset_name, metric, drop_out_ratio = 0.1, save_total_limit = 5)

    # Create folders in case report errors
    dataset_path = dataset_name.replace("/","-")
    os.makedirs("./training_outputs/{}".format(dataset_path), exist_ok=True)
    os.makedirs("./results/{}".format(dataset_path), exist_ok=True)

    # Parameter settings in the MAM paper (page.14) as DEFAULT parameter values if not specified in command-line args
    # Note: Paper's setting may not exactly apply to our case due to base model difference
    lr_default = [5e-5] if args.task in ["xsum","wmt16"] else [1e-4]
    warmup_ratio_default = [0] if args.task in ["xsum","wmt16"] else [0.06]
    label_smoothing_factor_default = [0.1] if args.task in ["xsum","wmt16"] else [0]
    weight_decay_default = [0.01] if args.task == "xsum" else [0.1]
    max_grad_norm_default = [0.1] if args.task == "xsum" else [1]

    # Set parameters - Use paper setting as default if not specified in args
    search_space = {
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate" : args.learning_rate if args.learning_rate else lr_default,
        "warmup_ratio" : args.warmup_ratio if args.warmup_ratio else warmup_ratio_default,
        "label_smoothing_factor" : args.label_smoothing_factor if args.label_smoothing_factor else label_smoothing_factor_default,
        "weight_decay" : args.weight_decay if args.weight_decay else weight_decay_default,
        "max_grad_norm" : args.max_grad_norm if args.max_grad_norm else max_grad_norm_default,
    }
    print("\nParameter Search Space:\n{}\n".format(search_space))

    # Train and save the best model hyperparameter set for the chosen adapter architecture that achives highest evaluation score (like accuracy)
    # Example argument: python model.py -T sst2 -A mam -P 0.1 -B 0.5 -s 4 -bs 32 64 -lr 1e-4 5e-5 -ep 3 5
    # Example output: best paramter set stored as ./results/stanfordnlp-sst2/stanfordnlp-sst2__mam_budget_0-1_alloc_0-5_scaling_4-0__{"accuracy": 0.9174}.json
    # Inside the json file: {"best_eval_scores": {"accuracy": 0.9174}, "adapter_setting": {"adapter": "mam", "par_bottleneck_size": 512.0, "bottleneck_size": 30.0, 
    #                        "scaling": 4.0, "reduction_factor": 1.5, "param_budget": 0.09148, "budget_alloc": 0.06}, "best_params": {"learning_rate": 0.0001, 
    #                        "per_device_train_batch_size": 32, "num_train_epochs": 1, "warmup_ratio": 0.06, "label_smoothing_factor": 0, "weight_decay": 0.1, "max_grad_norm": 1}}
    global eval_metrics_recorder
    eval_metrics_recorder = [] # Record evaluation metric values for each epoch
    # Start hyperparameter search
    best_params = trainer.hyperparameter_search(**hyperparameter_search_settings(search_space))
    # Search completed. Get best corresponding eval scores to the best trail
    best_eval_scores = {}
    for score_dic in eval_metrics_recorder:
        if round(sum(list(score_dic.values())),4) == round(best_params[1],4):
            best_eval_scores = score_dic
    if best_eval_scores == {}: best_eval_scores = {"score": round(best_params[1],4)} # In case any errors occur, go for default one
    print("\nBest eval performance: {}\nBest model parameter setting: {}\n".format(best_eval_scores, best_params))
    best_setting_save_path = "./results/{}/{}__{}__{}.json".format(dataset_path, dataset_path, adapter_name, best_eval_scores)
    print("Best setting saved to {}\n".format(best_setting_save_path))
    # Save the best model parameter settings for the target adapter architecture to local
    with open(best_setting_save_path, "w") as file:
        json.dump({"best_eval_scores":best_eval_scores, "adapter_setting":adapter_setting, "best_params":best_params[2]}, file)
