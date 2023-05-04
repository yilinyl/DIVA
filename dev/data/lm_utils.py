import torch
from transformers import AutoTokenizer, AutoModel

ESM_MODEL = "facebook/esm2_t12_35M_UR50D"

def init_pretrained_lm(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    return tokenizer, model


def calc_esm_emb(seq, tokenizer, model, clip=True):
    device = model.device
    with torch.no_grad():
        inputs = tokenizer(seq, return_tensors='pt').to(device)
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.squeeze(0)

        if clip:
            return emb[1:-1].detach().cpu()  # ESM add two additiona dims at the start & end

        return emb.detach().cpu()
