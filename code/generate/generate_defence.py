
"""Gradient-guided Token Masking for inference-time visual prompt-injection defense."""

import torch

img_embeddings_captured = None


def generate_from_embeds(model, processor, inputs_embeds, max_new_tokens=200, verbose=False):
    """Autoregressively decode from prepared input embeddings."""
    device = model.device
    batch_size, seq_len, _ = inputs_embeds.shape
    if hasattr(processor, "tokenizer"):
        eos_token_id = processor.tokenizer.eos_token_id
    else:
        eos_token_id = processor.eos_token_id
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    generated_ids = []



    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True
        )
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    generated_ids.append(next_token_id)



    for step in range(max_new_tokens - 1):
        if (next_token_id == eos_token_id).all():
            break
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones((batch_size, 1), dtype=torch.long, device=device)
        ], dim=1)
        with torch.no_grad():
            outputs = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated_ids.append(next_token_id)
    output_ids = torch.cat(generated_ids, dim=1)
    if hasattr(processor, "tokenizer"):
        text = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    else:
        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return text


def get_phi3_imb_index(inputs, text_input, model, processor, mask_rate=5):
    """Locate Phi-3-Vision image tokens with the hidden-state gradient norm."""
    global img_embeddings_captured
    img_embeddings_captured = None
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()

    def vision_hook(module, args, output):
        global img_embeddings_captured
        img_embeddings_captured = output
        img_embeddings_captured.requires_grad_(True)
        img_embeddings_captured.retain_grad()
    handle = model.model.vision_embed_tokens.img_projection.register_forward_hook(vision_hook)



    outputs = model(**inputs, output_hidden_states=True)
    input_ids = inputs.input_ids[0]
    text_token_ids = processor.tokenizer.encode(" "+text_input, add_special_tokens=False)
    if len(text_token_ids) == 0:
        raise ValueError("Text input is empty or cannot be encoded.")
    first_text_token_id = text_token_ids[0]
    matches = (input_ids == first_text_token_id).nonzero(as_tuple=True)[0]
    target_index = matches[-1].item()
    last_hidden_state = outputs.hidden_states[-1]
    target_embedding = last_hidden_state[0, target_index, :]
    loss = torch.norm(target_embedding, p=2)
    model.zero_grad()
    loss.backward()


    grads = img_embeddings_captured.grad
    if len(grads.shape) == 3:
        saliency_map = torch.norm(grads, dim=-1)[0]
    else:
        saliency_map = torch.norm(grads, dim=-1)
    k = int(saliency_map.numel() * mask_rate / 100)
    if k > 0:
        _, top_indices = torch.topk(saliency_map, k=k)
        indices = top_indices.tolist()
    else:
        indices = []
    handle.remove()
    torch.cuda.empty_cache()
    return indices, img_embeddings_captured

def get_qwen_imb_index(inputs, text_input, model, processor, mask_rate=5):
    """Locate Qwen2-VL image tokens with the hidden-state gradient norm."""
    global img_embeddings_captured
    img_embeddings_captured = None
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()

    def vision_hook(module, args, output):
        global img_embeddings_captured
        img_embeddings_captured = output
        img_embeddings_captured.requires_grad_(True)
        img_embeddings_captured.retain_grad()
    handle = model.visual.register_forward_hook(vision_hook)


    outputs = model(**inputs, output_hidden_states=True)
    input_ids = inputs.input_ids[0]
    text_token_ids = processor.tokenizer.encode(text_input, add_special_tokens=False)
    first_text_token_id = text_token_ids[0]
    target_index = (input_ids == first_text_token_id).nonzero(as_tuple=True)[0][-1].item()
    last_hidden_state = outputs.hidden_states[-1]
    target_embedding = last_hidden_state[0, target_index, :]
    loss = torch.norm(target_embedding, p=2)
    model.zero_grad()
    loss.backward()


    grads = img_embeddings_captured.grad
    saliency_map = torch.norm(grads, dim=-1)
    if saliency_map.dim() > 1:
        saliency_map = saliency_map[0]
    k = int(saliency_map.numel() * mask_rate / 100)
    if k > 0:
        _, top_indices = torch.topk(saliency_map, k=k)
        indices = top_indices.tolist()
    else:
        indices = []
    handle.remove()
    torch.cuda.empty_cache()
    return indices, img_embeddings_captured

def get_llava_imb_index(inputs, text_input, model, processor, mask_rate=5):
    """Locate LLaVA image tokens with the hidden-state gradient norm."""
    global img_embeddings_captured
    img_embeddings_captured = None
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()

    def vision_hook(module, args, output):
        global img_embeddings_captured
        img_embeddings_captured = output
        img_embeddings_captured.requires_grad_(True)
        img_embeddings_captured.retain_grad()
    handle = model.multi_modal_projector.register_forward_hook(vision_hook)



    outputs = model(**inputs, output_hidden_states=True)
    input_ids = inputs.input_ids[0]
    text_token_ids = processor.tokenizer.encode(" "+text_input, add_special_tokens=False)
    first_text_token_id = text_token_ids[0]
    target_index = (input_ids == first_text_token_id).nonzero(as_tuple=True)[0][-1].item()
    last_hidden_state = outputs.hidden_states[-1]
    target_embedding = last_hidden_state[0, target_index, :]
    loss = torch.norm(target_embedding, p=2)
    model.zero_grad()
    loss.backward()


    grads = img_embeddings_captured.grad
    saliency_map = torch.norm(grads, dim=-1)
    if saliency_map.dim() > 1:
        saliency_map = saliency_map[0]
    k = int(saliency_map.numel() * mask_rate / 100)
    if k > 0:
        _, top_indices = torch.topk(saliency_map, k=k)
        indices = top_indices.tolist()
    else:
        indices = []
    handle.remove()
    torch.cuda.empty_cache()
    return indices, img_embeddings_captured

def generate_defence(model, processor, inputs, text_input, image_input, max_new_tokens=512, do_sample=False, max_rate = 0.05, mask_rate = 5):
    """Apply GTM by zeroing high-saliency image tokens before generation."""
    if "qwen" in str(model).lower():

        index, img_emb = get_qwen_imb_index(inputs,text_input,model,processor, mask_rate)
        drop_img_emb = img_emb.clone()
        drop_img_emb[index,:] = 0
        drop_img_emb = drop_img_emb.unsqueeze(0)



        input_ids = inputs["input_ids"]
        token_emb = model.get_input_embeddings()(input_ids)
        image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        mask = (input_ids == image_pad_id)

        def replace_image_embeds(token_emb, mask, adv_image_emb):
            """Insert masked visual embeddings at Qwen image-token positions."""
            batch, seq_len, hidden = token_emb.shape
            img_len = adv_image_emb.shape[1]
            new_embeds = []
            for b in range(batch):
                seq = []
                i = 0
                while i < seq_len:
                    if mask[b, i]:
                        seq.append(adv_image_emb[b])
                        while i < seq_len and mask[b, i]:
                            i += 1
                    else:
                        seq.append(token_emb[b, i].unsqueeze(0))
                        i += 1
                new_embeds.append(torch.cat(seq, dim=0))
            return torch.stack(new_embeds, dim=0)

        inputs_embeds = replace_image_embeds(token_emb, mask, drop_img_emb)
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            do_sample=True, max_new_tokens=512, temperature=0.2, top_p=0.9,
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    elif "llava" in str(model).lower():

        index, img_emb = get_llava_imb_index(inputs,text_input,model,processor, mask_rate)
        drop_img_emb = img_emb.clone()
        drop_img_emb = drop_img_emb[0]
        drop_img_emb[index,:] = 0
        drop_img_emb = drop_img_emb.unsqueeze(0)


        input_ids = inputs["input_ids"].to(model.device)
        token_emb = model.get_input_embeddings()(input_ids)
        image_pad_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        mask = (input_ids == image_pad_id)

        def replace_image_embeds(token_emb, mask, adv_image_emb):
            """Insert masked visual embeddings at LLaVA image-token positions."""
            batch, seq_len, hidden = token_emb.shape
            img_len = adv_image_emb.shape[1]
            new_embeds = []
            for b in range(batch):
                seq = []
                i = 0
                while i < seq_len:
                    if mask[b, i]:
                        seq.append(adv_image_emb[b])
                        while i < seq_len and mask[b, i]:
                            i += 1
                    else:
                        seq.append(token_emb[b, i].unsqueeze(0))
                        i += 1
                new_embeds.append(torch.cat(seq, dim=0))
            return torch.stack(new_embeds, dim=0)

        inputs_embeds = replace_image_embeds(token_emb, mask, drop_img_emb)
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            do_sample=True, max_new_tokens=512, temperature=0.2, top_p=0.9,
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    elif "phi" in str(model).lower():

        index, img_emb = get_phi3_imb_index(inputs, text_input, model, processor, mask_rate)
        drop_img_emb = img_emb.clone().detach()
        if drop_img_emb.dim() == 2:
            drop_img_emb = drop_img_emb.unsqueeze(0)
        drop_img_emb[0, index, :] = 0




        input_ids = inputs["input_ids"]
        mask = (input_ids <= 0)
        safe_input_ids = input_ids.clone()
        safe_input_ids[mask] = 0
        token_emb = model.model.embed_tokens(safe_input_ids)

        def replace_image_embeds(token_emb, mask, adv_image_emb):
            """Insert masked visual embeddings at Phi-3 visual-token positions."""
            batch, seq_len, hidden = token_emb.shape
            if adv_image_emb.shape[0] != batch:
                 adv_image_emb = adv_image_emb.expand(batch, -1, -1)
            new_embeds = []
            for b in range(batch):
                seq = []
                i = 0
                img_inserted = False
                while i < seq_len:
                    if mask[b, i]:
                        if not img_inserted:
                            seq.append(adv_image_emb[b])
                            img_inserted = True
                        while i < seq_len and mask[b, i]:
                            i += 1
                    else:
                        seq.append(token_emb[b, i].unsqueeze(0))
                        i += 1
                new_embeds.append(torch.cat(seq, dim=0))
            return torch.stack(new_embeds, dim=0)

        inputs_embeds = replace_image_embeds(token_emb, mask, drop_img_emb)
        generated_text = generate_from_embeds(model=model, processor=processor, inputs_embeds=inputs_embeds, max_new_tokens=512)
    return generated_text
