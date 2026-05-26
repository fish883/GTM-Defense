"""Generation entry point for GTM and the undefended baseline."""

from generate_defence import generate_defence
from generate_wo import generate_wo

def generate(model, processor, inputs, text_input, image_input, defence_method):
    """Dispatch generation to Gradient-guided Token Masking or the baseline."""
    if defence_method == "our":
        return generate_defence(model, processor, inputs, text_input, image_input)
    elif defence_method == "wo":
        return generate_wo(model, processor, inputs, text_input, image_input)

