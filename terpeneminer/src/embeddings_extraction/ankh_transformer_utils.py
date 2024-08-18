"""This script contains utils for Ankh embeddings extraction"""
import ankh  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore


def get_model_and_tokenizer(
    model_name: str,
) -> tuple:
    """
    This function returns bert model and batch converter (basically a tokenizer) based on the model name
    :param model_name: model name
    :return: a pair of the bert protein model and its tokenizer
    """
    assert model_name in {
        "ankh_large",
        "ankh_base",
        "ankh_tps",
    }, f"Model {model_name} is not supported. Choose between ankh_tps, ankh_large and ankh_base"
    if model_name == "ankh_large":
        model, tokenizer = ankh.load_large_model()
    elif model_name == "ankh_tps":
        model, tokenizer = ankh.load_base_model(generation=True)
        model.load_state_dict(
            torch.load("data/plm_checkpoints/tps_ankh_lr=5e-05_bs=32.pth")[
                "model_state_dict"
            ],
            strict=False,
        )
    else:
        model, tokenizer = ankh.load_base_model()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def compute_embeddings(
    bert_model,
    tokenizer: ankh.models.ankh_transformers.AutoTokenizer,
    input_seqs: list[str],
) -> tuple[np.ndarray, list]:
    """
    This function computes bert_model embeddings of protein sequences
    :param bert_model: trained Protein model
    :param tokenizer: tokenizer
    :param input_seqs: list of protein sequences stored
    :return: numpy array with average protein embeddings and array with a sequence of amino acid embeddings
    """
    bert_model.eval()
    protein_sequences = [list(seq) for seq in input_seqs]
    outputs = tokenizer.batch_encode_plus(
        protein_sequences,
        add_special_tokens=True,
        padding=True,
        is_split_into_words=True,
        return_tensors="pt",
    )
    if torch.cuda.is_available():
        outputs["input_ids"] = outputs["input_ids"].to(device="cuda", non_blocking=True)
        outputs["attention_mask"] = outputs["attention_mask"].to(
            device="cuda", non_blocking=True
        )
    with torch.no_grad():
        try:
            embeddings_batch_raw = (
                bert_model(
                    input_ids=outputs["input_ids"],
                    attention_mask=outputs["attention_mask"],
                )
                .last_hidden_state.cpu()
                .numpy()
            )
        except ValueError:
            embeddings_batch_raw = (
                bert_model(
                    input_ids=outputs["input_ids"],
                    attention_mask=outputs["attention_mask"],
                    decoder_input_ids=outputs["input_ids"],
                    decoder_attention_mask=outputs["attention_mask"],
                )
                .encoder_last_hidden_state.cpu()
                .numpy()
            )

    # masks to extract relevant tokens from the batch embs
    masks = outputs["attention_mask"].cpu().numpy().astype(bool)
    encodings_batch = []
    encoding_seqs_batch = []
    for i, embs_per_tokens in enumerate(embeddings_batch_raw):
        embs_per_tokens = embs_per_tokens[masks[i]]
        encodings_batch.append(embs_per_tokens.mean(0))
        encoding_seqs_batch.append(embs_per_tokens)
    encodings_np_batch = np.array(encodings_batch)
    return encodings_np_batch, encoding_seqs_batch
