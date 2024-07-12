import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers import LlavaForConditionalGeneration, CLIPVisionModel,AutoModelForCausalLM

from transformers import CLIPImageProcessor, AutoTokenizer
from transformers import LlavaProcessor
from transformers import CLIPVisionConfig, AutoConfig, LlavaConfig

from PIL import Image

class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class CustomLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config: LlavaConfig):
        super(LlavaForConditionalGeneration, self).__init__(config)  # Use super() to call the parent class constructor
        self.vision_tower = CLIPVisionModel.from_pretrained(config.vision_config._name_or_path,
                                                            torch_dtype=torch.float32)

        self.multi_modal_projector = LlavaMultiModalProjector(config).to(torch.float32)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_pretrained(config.text_config._name_or_path,
                                                                   torch_dtype=torch.float32, 
                                                                   #attn_implementation=config._attn_implementation
                                                                   )
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()



class ImageTextInstructionFollowingDataset(torch.utils.data.Dataset):
    def __init__(self, data, image_folder_path=None):
        self.data = data
        self.image_folder_path = image_folder_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        conversation = self.data["conversations"][index]
        image_path = self.data["image"][index]
        if self.image_folder_path is not None:
            image_path = f"{self.image_folder_path}/{image_path}"
        img = Image.open(image_path)
        instruction = conversation[0]["value"]
        answer = conversation[1]["value"]

        instruction_answer = instruction + "\n\nAnswer:" + answer
        return instruction_answer, img


class MyCustomDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        instruction_answers = [item[0] for item in examples]
        images = [item[1] for item in examples]
        
        tokenizer_outputs = self.processor.tokenizer(
            instruction_answers, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids = tokenizer_outputs['input_ids']
        attention_mask = tokenizer_outputs['attention_mask']

        pixel_values = self.processor.image_processor(
            #images=images, return_tensors="pt", padding=True
            images=images, return_tensors="pt"
        )["pixel_values"]

        #if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
        #    raise ValueError("NaN or Inf found in input_ids during batching")

        #if torch.isnan(attention_mask).any() or torch.isinf(attention_mask).any():
        #    raise ValueError("NaN or Inf found in attention_mask during batching")

        #if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
        #    raise ValueError("NaN or Inf found in pixel_values during batching")

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'labels': input_ids
        }
        return batch



def get_processor(image_processor_id, tokenizer_id, image_token):
    image_processor = CLIPImageProcessor.from_pretrained(image_processor_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    image_token_to_add = image_token
    tokenizer.add_tokens([image_token_to_add])

    #image_token_index = tokenizer(image_token_to_add)['input_ids'][0]

    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    
    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
    return processor


def get_model(vision_tower_id, language_model_id, tokenizer, image_token):
    vision_config = CLIPVisionConfig.from_pretrained(vision_tower_id)
    vision_config._name_or_path = vision_tower_id

    text_config = AutoConfig.from_pretrained(language_model_id)

    configuration = LlavaConfig(vision_config, text_config)
    configuration._attn_implementation = "flash_attention_2"
    configuration.image_token_index = tokenizer(image_token)['input_ids'][0]
    configuration.pad_token_id = tokenizer.pad_token_id

    print(f"hahahahaha {tokenizer(image_token)['input_ids'][0]}")
    print(f"hahahahaha {tokenizer.pad_token}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomLlavaForConditionalGeneration(configuration).to(device)
    return model


    


