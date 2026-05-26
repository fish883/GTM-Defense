import torch
from tqdm import tqdm
import random
import gc
from pathlib import Path
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import seaborn as sns

from torchvision import transforms 

from PIL import Image,ImageChops
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
import torchvision.transforms as T

topil = T.ToPILImage()
totensor = T.ToTensor()

def normalize(images,device):
    """Normalize image tensors with Qwen2-VL image statistics."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images,device):
    """Restore normalized images to pixel space."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])


class Attacker:
    """Image adversarial attacker for Qwen2-VL."""

    def __init__(self, args, model, processor, data, device=None, is_rtp=False, image_processor=None):

        self.args = args
        self.model = model
        self.processor= processor
        self.device = device or getattr(args, "device", "cuda:0")
        self.is_rtp = is_rtp

        self.data = data
        self.num_targets = len(data)

        self.loss_buffer = []
        self.loss_steps = []

        self.model.eval()
        self.model.requires_grad_(False)

        self.image_processor = image_processor

    def attack_constrained(self, data, img, batch_size = 8, num_iter=2000, alpha = 1/255, constrain = "l8" ):
        """Iteratively generate adversarial noise under a given constraint."""
        x = img.clone()
        if constrain.startswith("l"):
            # l* constraints bound full-image pixel perturbations.
            if constrain[1:] == "n":
                num = 255
            else:
                num = int(constrain[1:])
            adv_noise = torch.rand_like(img).to(self.device) * 2 * num/255 - num/255     
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
        else:
            # Non-l* constraints use a learnable local patch.
            num = int(constrain[1:])
            adv_noise = torch.rand(3, num, num)
        adv_noise = adv_noise.to(self.device)
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        batch_grad = torch.zeros_like(adv_noise)
        avg_loss = 0

        for t in tqdm(range(num_iter + 1)):

            # Sample target examples for the current iteration.
            batch_data = random.sample(data, batch_size)

            if constrain.startswith("l"):
                x_adv = x + adv_noise
            elif constrain.startswith("s"):
                x_adv = x.clone()
                x_adv[0, :,0:num, 0:num] = adv_noise
            else:
                x_adv = x.clone()
                index = torch.randint(0, 448-num-1, (2,))
                x_adv[0, :,index[0]:index[0]+num,index[1]:index[1]+num] = adv_noise

            x_adv_copy = x_adv.clone().detach()
            x_adv = normalize(x_adv,self.device)            
            target_loss, input,attention,image = self.attack_loss(batch_data, x_adv)
            avg_loss = target_loss.item() + avg_loss
            
            target_loss.backward()

            # Accumulate gradients and update the perturbation periodically.
            batch_grad = batch_grad + adv_noise.grad.detach()
            if t % 8 == 0:   
                if constrain.startswith("l"):   
                    num = int(constrain[1:])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                    adv_noise.data = (adv_noise.data - alpha * batch_grad.detach().sign()).clamp(-num/255, num/255)
                    adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
                else:
                    adv_noise.data = (adv_noise.data - alpha * batch_grad.detach().sign()).clamp(0, 1)
                batch_grad =  torch.zeros_like(adv_noise)
                self.loss_buffer.append(avg_loss)
                self.loss_steps.append(t)
                avg_loss = 0
            adv_noise.grad.zero_()
            self.model.zero_grad()

            target_loss =torch.tensor([0]).to(self.device)
            torch.cuda.empty_cache()

            if t % 80 == 0:
                # Save the loss curve periodically for monitoring.
                self.plot_loss()

            if t % 1000 == 0:
                # Save intermediate adversarial images periodically.
                x_adv = denormalize(x_adv,self.device)

                adv_img_prompt = x_adv.detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                output_dir = Path("middle")
                output_dir.mkdir(parents=True, exist_ok=True)
                save_image(adv_img_prompt, output_dir / f"bad_prompt_temp_{t}.bmp")
            gc.collect()
            torch.cuda.empty_cache()


        return adv_img_prompt

    def plot_loss(self): 
        """Save the current attack loss curve."""
        if len(self.loss_buffer) == 1:
            return

        sns.set_theme()
        plt.plot(self.loss_steps[1:], self.loss_buffer[1:], label='Target Loss')

        plt.title('Loss Plot')
        plt.xlabel('t')
        plt.ylabel('Loss')

        plt.legend(loc='best')
        output_dir = Path(self.args.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"loss_curve_{self.args.constrain}.png")
        plt.clf()

        torch.save(self.loss_buffer, 'loss')

    def attack_loss(self, batch_data, images):
        """Compute cross-entropy loss over the target response span."""

        processor = self.processor

        # Convert image tensors to the Qwen2-VL visual token layout.
        qwen_image = self.image_ori_to_qwen(images).to(self.device)
        
        image_grid_thw = torch.tensor([1, 448 // processor.image_processor.patch_size,  448 // processor.image_processor.patch_size], device = self.device)
        input_ids_list = []
        for data in batch_data:
            # Build input tokens with image, prompt, and target response.
            input_ids, prompt_process = self.get_input_ids(data[0],Image.new('RGB', (448, 448), (255, 255, 255)), data[1])
            input_ids = input_ids.to(self.device)
            input_ids_list.append(input_ids)

        max_length = max(len(ids) for ids in input_ids_list)

        padded_input_ids = []
        attention_mask = []
        for ids in input_ids_list:
            # Left-pad sequences to a uniform batch length and build masks.
            padding_length = max_length - len(ids)
            padded_ids = torch.cat([torch.full((padding_length,), processor.tokenizer.pad_token_id, device=ids.device), ids])
            attention_mask_tensor = torch.cat([torch.zeros(padding_length, device=ids.device), torch.ones(len(ids), device=ids.device)])
            padded_input_ids.append(padded_ids)
            attention_mask.append(attention_mask_tensor)
        
        input_ids_list = torch.stack(padded_input_ids)
        attention_mask = torch.stack(attention_mask)


        output = self.model(
            input_ids=input_ids_list[0].unsqueeze(0),
            attention_mask=attention_mask[0].unsqueeze(0),
            pixel_values=qwen_image,
            image_grid_thw=image_grid_thw.unsqueeze(0),
        )

        output_logits = output['logits']
        
        crit = torch.nn.CrossEntropyLoss(reduction='none')
        target_loss_list = []

        
        for id in range(output_logits.shape[0]):
            # Apply supervised loss only on target response positions.
            loss_slice = slice(self._target_slice.start - 1, self._target_slice.stop - 1)
            

            valid_output_logits = output_logits[id - 0][attention_mask[id] == 1]
            valid_input_ids = input_ids_list[id][attention_mask[id] == 1]

            text = self.processor.tokenizer.decode(valid_input_ids[self._target_slice], skip_special_tokens=True)
            ids = torch.argmax(valid_output_logits, dim=-1) 
            text = self.processor.tokenizer.decode(ids, skip_special_tokens=True)
            text = self.processor.tokenizer.decode(ids[loss_slice], skip_special_tokens=True)

            target_loss = crit(valid_output_logits[loss_slice, :], valid_input_ids[self._target_slice])
            target_loss = target_loss.mean(dim=-1)
            target_loss_list.append(target_loss)

        stacked_target_loss = torch.stack(target_loss_list)

        total_loss = torch.sum(stacked_target_loss)

        total_loss = total_loss/1

        return total_loss,input_ids_list[0].unsqueeze(0),attention_mask[0].unsqueeze(0),qwen_image

    def average_pool_manual(self, avg_grad, grid_size=4):
        """Average-pool gradients and resize them to the original shape."""
        pooled = F.avg_pool2d(avg_grad, kernel_size=grid_size, stride=grid_size)
        
        modified_avg_grad = F.interpolate(pooled, size=avg_grad.shape[2:], mode='nearest')

        return modified_avg_grad

    def image_qwen_to_ori(self, images, image_grid_thw):
        """Convert Qwen2-VL visual token layout back to image tensors."""
        assert images.dim() == 2, "images must be 2D tensor"

        processor = self.processor
        temporal_patch_size = processor.image_processor.temporal_patch_size
        model_patch_size = processor.image_processor.patch_size
        merge_size = processor.image_processor.merge_size

        grid_t, grid_h, grid_w = image_grid_thw
        images = images.reshape(
            -1, grid_h // merge_size, grid_w // merge_size, 
            merge_size, merge_size, 3, 
            temporal_patch_size, model_patch_size, model_patch_size
        )
        images = images.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
        images = images.reshape(-1, 3, grid_h * model_patch_size, grid_w * model_patch_size)

        return images
    
    def image_ori_to_qwen(self, images):
        """Convert image tensors to the Qwen2-VL visual token layout."""

        
        assert images.dim() == 4, "images must be 4D tensor"

        processor = self.processor
        temporal_patch_size = processor.image_processor.temporal_patch_size
        model_patch_size = processor.image_processor.patch_size
        merge_size = processor.image_processor.merge_size

        if images.shape[0] == 1:
            images = images.repeat(temporal_patch_size, 1, 1, 1)
        channel = images.shape[1]
        grid_t = images.shape[0] // temporal_patch_size
        grid_h, grid_w = 448 // model_patch_size, 448 // model_patch_size
        images = images.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            model_patch_size,
            grid_w // merge_size,
            merge_size,
            model_patch_size,
        )
        images = images.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        images = images.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * model_patch_size * model_patch_size
        )

        return images
    

    def load_image(self,image_file, input_size=448, max_num=12):
        """Load and normalize a single image."""
        image_mean, image_std = self.config['preprocessing']['image_mean'], self.config['preprocessing']['image_std']
        if isinstance(image_file, str):
            image = Image.open(image_file).convert('RGB')
        else:
            image = image_file.convert('RGB')

        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.ToTensor(),
            T.Normalize(mean=image_mean, std=image_std)
        ])

        pixel_values = transform(image).unsqueeze(0).to(self.pipeline['model'].dtype).to(self.device)
        return pixel_values 

    def get_input_ids(self,prompt,image, target):
        """Build chat-template inputs and record key token spans."""

        eos_ids = self.processor.tokenizer.eos_token_id
        messages = []
        messages.append({
            "role": "user",
            "content":""
        })
        toks = self.qwen_tokenizer(messages=messages, with_content=False)
        self._user_role_slice = slice(None, len(toks))

        messages[0]['content']= [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": prompt},
        ]
        toks = self.qwen_tokenizer(messages=messages, with_content=True)
        self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks) - 1))
        self._control_slice = self._goal_slice

        messages.append({
            "role": "assistant",
            "content":""
        })
        toks = self.qwen_tokenizer(messages=messages, with_content=False)
        self._assistant_role_slice = slice(self._control_slice.stop, len(toks))
        messages[1]['content']= [
            {"type": "text", "text": target},
        ]

        toks = self.qwen_tokenizer(messages=messages, with_content=True)
        target_end_indices = torch.nonzero(toks[self._assistant_role_slice.stop:]== eos_ids)[0].item()
        self._target_slice = slice(self._assistant_role_slice.stop, self._assistant_role_slice.stop + target_end_indices)
        self._loss_slice = slice(self._assistant_role_slice.stop - 1, self._assistant_role_slice.stop + target_end_indices - 1)
        
        prompt = messages


        toks = self.qwen_tokenizer(messages=prompt)
        input_ids = toks[:self._target_slice.stop]

        return input_ids, prompt


    def qwen_tokenizer(self,messages, with_content=True):
        """Generate text and visual input tokens with the Qwen2-VL processor."""
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        if with_content is False:
            text = text[:-len("<|im_end|>\n")]

        text = [text]
        image_inputs, video_inputs = process_vision_info(messages)
        toks = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).input_ids[0]

        return toks
