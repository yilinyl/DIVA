import numpy as np
import torch


def extract_text_emb(text_encoder, device, pheno_loader):
    text_encoder.eval()
    all_pheno_emb_list = []

    with torch.no_grad():
        for idx, batch_pheno in enumerate(pheno_loader):
            # pheno_input_dict = load_input_to_device(batch_pheno, device)
            pheno_input_dict = batch_pheno.to(device)
            pheno_embs = text_encoder(
                pheno_input_dict['input_ids'],
                attention_mask=pheno_input_dict['attention_mask'],
                token_type_ids=pheno_input_dict['token_type_ids'],
                # token_type_ids=torch.zeros(pheno_input_dict['input_ids'].size(), dtype=torch.long, device=self.device),
                output_attentions=False,
                output_hidden_states=True,
                return_dict=None
                ).hidden_states[-1]  # n_var, max_pos_pheno_length, pheno_emb_dim
            batch_size = pheno_input_dict['input_ids'].shape[0]
            pheno_embs = torch.stack([pheno_embs[i, pheno_input_dict['attention_mask'][i, :].bool()].mean(dim=0) for i in range(batch_size)], dim=0)
            all_pheno_emb_list.append(pheno_embs.detach().cpu().numpy())
        all_pheno_embs = np.concatenate(all_pheno_emb_list, 0)        

    return all_pheno_embs

