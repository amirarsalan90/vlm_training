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
        super(LlavaForConditionalGeneration, self).__init__(config)  
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



def get_processor(image_processor_id, tokenizer_id, image_token):
    image_processor = CLIPImageProcessor.from_pretrained(image_processor_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    image_token_to_add = image_token
    tokenizer.add_tokens([image_token_to_add])

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomLlavaForConditionalGeneration(configuration).to(device)
    return model


    


