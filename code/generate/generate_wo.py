"""Undefended generation baseline used for comparison with GTM."""

def generate_wo(model, processor, inputs, text_input,image_input, max_new_tokens=512, do_sample=False):
    """Generate a response directly from the original multimodal inputs."""
    generated_ids = model.generate(**inputs, do_sample=True, max_new_tokens=512, temperature=0.2, top_p=0.9,)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output
