import math
from abc import ABC, abstractmethod

import clip
import numpy as np
import PIL
import PIL.Image
import torch
from PIL.Image import Image
from torch.nn.functional import softmax
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToPILImage,
    ToTensor,
)
from tqdm import tqdm

from src.analysis.neuroai.neuroai.models.pytorch.clip.imagenet_classes import (
    imagenet_classes,
)
from src.analysis.neuroai.neuroai.models.pytorch.clip.imagenet_templates import (
    imagenet_templates,
)
from src.analysis.neuroai.neuroai.models.wrappers.base import AbstractModel

from ..pytorch.clip.imagenet_classes import imagenet_classes
from ..pytorch.clip.imagenet_templates import imagenet_templates
from .base import AbstractModel


def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def undo_default_preprocessing(images):
    """Convenience function: undo standard preprocessing."""

    assert type(images) is torch.Tensor
    default_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device())
    default_std = torch.Tensor([0.229, 0.224, 0.225]).to(device())

    images *= default_std[None, :, None, None]
    images += default_mean[None, :, None, None]

    return images


class PytorchModel(AbstractModel):

    def __init__(self, model, model_name, *args):
        self.model = model
        self.model_name = model_name
        self.args = args
        try:
            self.model.to(device())
        except:
            #  Accelerate (via device_map="auto") is already handling device placement.
            pass

    def to_numpy(self, x):
        if x.is_cuda:
            return x.detach().cpu().numpy()
        else:
            return x.numpy()

    def softmax(self, logits):
        assert type(logits) is np.ndarray

        softmax_op = torch.nn.Softmax(dim=1)
        softmax_output = softmax_op(torch.Tensor(logits))
        return self.to_numpy(softmax_output)

    def forward_batch(self, images):
        assert type(images) is torch.Tensor

        self.model.eval()
        logits = self.model(images)
        return self.to_numpy(logits)


class TwoStreamDVDModel(PytorchModel):
    "This class inherits PytorchModel to adapt two stream dvd models."

    def __init__(self, model, model_name, *args):
        super().__init__(model, model_name, args)

    def forward_batch(self, x_1, x_2=None):
        self.model.eval()
        logits = self.model(x_1, x_2)
        return self.to_numpy(logits)


class PyContrastPytorchModel(PytorchModel):
    """
    This class inherits PytorchModel class to adapt model validation for Pycontrast pre-trained models from
    https://github.com/HobbitLong/PyContrast
    """

    def __init__(self, model, classifier, model_name, *args):
        super().__init__(model, model_name, args)
        self.classifier = classifier
        self.classifier.to(device())

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()
        self.classifier.eval()
        feat = self.model(images, mode=2)
        output = self.classifier(feat)
        return self.to_numpy(output)


class ViTPytorchModel(PytorchModel):

    def __init__(self, model, model_name, img_size=(384, 384), *args):
        self.img_size = img_size
        super().__init__(model, model_name, args)

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()

        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0)).to(device())

        logits = self.model(images)
        return self.to_numpy(logits)

    def preprocess(self):
        # custom preprocessing from:
        # https://github.com/lukemelas/PyTorch-Pretrained-ViT

        return Compose(
            [
                Resize(self.img_size),
                ToTensor(),
                Normalize(0.5, 0.5),
            ]
        )


class ClipPytorchModel(PytorchModel):

    def __init__(self, model, model_name, *args):
        super().__init__(model, model_name, *args)
        self.zeroshot_weights = self._get_zeroshot_weights(
            imagenet_classes, imagenet_templates
        )

    def _get_zeroshot_weights(self, class_names, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for class_name in tqdm(class_names):
                texts = [
                    template.format(class_name) for template in templates
                ]  # format with class
                texts = clip.tokenize(texts).to(device())  # tokenize
                class_embeddings = self.model.encode_text(
                    texts
                )  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(
                device()
            )

        return zeroshot_weights

    def preprocess(self):
        n_px = self.model.visual.input_resolution
        return Compose(
            [
                Resize(n_px, interpolation=PIL.Image.BICUBIC),
                CenterCrop(n_px),
                # lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def forward_batch(self, images):
        assert type(images) is torch.Tensor

        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0)).to(device())

        self.model.eval()

        image_features = self.model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ self.zeroshot_weights
        return self.to_numpy(logits)


import numpy as np
import torch
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToPILImage,
    ToTensor,
)
from tqdm import tqdm


class OpenCLIPPytorchModel(PytorchModel):

    def __init__(self, model, tokenizer, preprocess_fn, model_name, *args):
        super().__init__(model, model_name, *args)
        self.tokenizer = tokenizer
        self.preprocess_fn = preprocess_fn  # open_clip's preprocess
        self.zeroshot_weights = self._get_zeroshot_weights(
            imagenet_classes, imagenet_templates
        )

    def _get_zeroshot_weights(self, class_names, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for class_name in tqdm(class_names):
                texts = [template.format(class_name) for template in templates]
                tokens = self.tokenizer(texts).to(device())
                text_features = self.model.encode_text(tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                class_embedding = text_features.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            return torch.stack(zeroshot_weights, dim=1).to(device())

    def preprocess(self):
        return self.preprocess_fn

    def forward_batch(self, images):
        assert isinstance(images, torch.Tensor)
        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(img)) for img in images]
        images = torch.stack(images).to(device())

        self.model.eval()
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ self.zeroshot_weights
        return self.to_numpy(logits)


class EfficientNetPytorchModel(PytorchModel):

    def __init__(self, model, model_name, *args):
        super().__init__(model, model_name, *args)

    def preprocess(self):
        normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        img_size = 475
        crop_pct = 0.936
        scale_size = int(math.floor(img_size / crop_pct))
        return Compose(
            [
                Resize(scale_size, interpolation=PIL.Image.BICUBIC),
                CenterCrop(img_size),
                ToTensor(),
                normalize,
            ]
        )

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()

        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0)).to(device())

        logits = self.model(images)
        return self.to_numpy(logits)


class SwagPytorchModel(PytorchModel):

    def __init__(self, model, model_name, input_size, *args):
        super().__init__(model, model_name, *args)
        self.input_size = input_size

    def preprocess(self):
        normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        return Compose(
            [
                Resize(self.input_size, interpolation=PIL.Image.BICUBIC),
                CenterCrop(self.input_size),
                ToTensor(),
                normalize,
            ]
        )

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()
        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0)).to(device())
        logits = self.model(images)
        return self.to_numpy(logits)


# * VLMs
# class QwenVLPytorchModel(PytorchModel):
#     def __init__(self, model, processor, model_name, *args):
#         super(QwenVLPytorchModel, self).__init__(model, model_name, *args)
#         self.processor = processor
#         self.zeroshot_weights = self._get_zeroshot_weights(imagenet_classes, imagenet_templates)
#         self.device = device()

#     def _get_zeroshot_weights(self, class_names, templates, batch_size=512):
#         """Generate zero-shot classification weights using QWEN's text encoder
#         by processing texts in batches for efficiency.
#         """
#         all_class_embeddings = []
#         with torch.no_grad():
#             for class_name in tqdm(class_names, desc="Generating Zero-Shot Weights"):
#                 texts = [template.format(class_name) for template in templates]

#                 # Prepare conversations for all templates for the current class
#                 conversations = [[{"role": "user", "content": [{"type": "text", "text": text}]}] for text in texts]

#                 # Batch process texts
#                 text_embeddings_for_class = []
#                 for i in range(0, len(conversations), batch_size):
#                     batch_conversations = conversations[i:i + batch_size]

#                     # Apply chat template and tokenize for the batch
#                     inputs = self.processor.apply_chat_template(
#                         batch_conversations,
#                         add_generation_prompt=False,
#                         tokenize=True,
#                         return_dict=True,
#                         return_tensors="pt",
#                         padding=True # Crucial for batching
#                     ).to(device())

#                     outputs = self.model(**inputs, output_hidden_states=True)
#                     batch_text_embeddings = outputs.hidden_states[-1].mean(dim=1)
#                     text_embeddings_for_class.append(batch_text_embeddings)

#                 # Stack and average embeddings across templates for the class
#                 class_embedding = torch.cat(text_embeddings_for_class).mean(dim=0, keepdim=True)
#                 class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
#                 all_class_embeddings.append(class_embedding)

#             zeroshot_weights = torch.cat(all_class_embeddings, dim=0).to(device())
#         return zeroshot_weights

#     def preprocess(self):
#         """Return identity function as QWEN processor handles preprocessing"""
#         return lambda x: x

#     def _extract_image_features(self, pil_images, batch_size=16):
#         """Extract image features using QWEN vision encoder by processing images in batches."""
#         all_image_features = []
#         with torch.no_grad():
#             for i in tqdm(range(0, len(pil_images), batch_size), desc="Extracting Image Features"):
#                 batch_pil_images = pil_images[i:i + batch_size]

#                 # Prepare conversations for the batch
#                 batch_conversations = []
#                 for image in batch_pil_images:
#                     conversation = [{
#                         "role": "user",
#                         "content": [
#                             {"type": "image", "image": image},
#                             {"type": "text", "text": ""} # Empty text for feature extraction
#                         ]
#                     }]
#                     batch_conversations.append(conversation)

#                 # Apply chat template and tokenize for the batch
#                 # Qwen2VL's processor handles image input directly when passed in the content list
#                 inputs = self.processor.apply_chat_template(
#                     batch_conversations,
#                     add_generation_prompt=False,
#                     tokenize=True,
#                     return_dict=True,
#                     return_tensors="pt",
#                     padding=True # Important for batching
#                 ).to(device())

#                 outputs = self.model(**inputs, output_hidden_states=True)

#                 # Extract vision features from hidden states.
#                 # The number of vision tokens (e.g., 576) might vary based on image size and model specifics.
#                 # You might need to verify the exact slice for vision tokens.
#                 # A more robust way might be to identify tokens based on special vision tokens if available.
#                 vision_features = outputs.hidden_states[-1][:, :576, :] # Typical vision token count

#                 # Pool vision tokens
#                 image_features = vision_features.mean(dim=1)
#                 image_features /= image_features.norm(dim=-1, keepdim=True)
#                 all_image_features.append(image_features)

#         return torch.cat(all_image_features, dim=0)

#     def forward_batch(self, images):
#         """Forward pass for batch of images"""
#         assert isinstance(images, torch.Tensor)

#         images = undo_default_preprocessing(images)
#         pil_images = [ToPILImage()(image.cpu()) for image in images] # Ensure images are on CPU before converting to PIL

#         self.model.eval()

#         # Extract image features
#         image_features = self._extract_image_features(pil_images)

#         # Compute similarity with zero-shot weights
#         # Reshape for matrix multiplication
#         image_features = image_features.squeeze(1) if image_features.dim() == 3 else image_features
#         zeroshot_weights = self.zeroshot_weights.squeeze(1) if self.zeroshot_weights.dim() == 3 else self.zeroshot_weights

#         try:
#             # Ensure both tensors are on the same device before multiplication
#             logits = 100. * image_features @ zeroshot_weights.T
#         except Exception as e:
#             print(f"Error during matrix multiplication: {e}")
#             import pdb;pdb.set_trace()

#         return self.to_numpy(logits)

import re

import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from tqdm import tqdm


class QwenVLPytorchModel(PytorchModel):
    def __init__(self, model, processor, model_name, *args, **kwargs):
        super().__init__(model, model_name, *args)
        self.processor = processor
        self.device = device()

        # Extract batch_size from kwargs or set a default
        self.batch_size = kwargs.pop("batch_size", 16)

        # Store class names and templates for direct questioning
        self.class_names = kwargs.get("class_names", imagenet_classes)
        self.templates = kwargs.get("templates", imagenet_templates)

        # Create the direct question prompt
        self._setup_question_prompt()

    def _setup_question_prompt(self):
        """Setup the question prompt template for direct questioning"""
        # Create class options string
        class_options = "\n".join(self.class_names[:])

        # Create the base prompt template (without USER/ASSISTANT formatting)
        self.question_template = (
            f"You are an ImageNet classifier. Choose exactly one class label from the list provided by the user."
            f"Respond with *only* that class name—no quotes, no punctuation, no additional "
            f"words. If uncertain, guess the most likely class. Never apologise or mention "
            f"uncertainty. Classify this image among the 1000 ImageNet classes:\n{class_options}\n"
            # 'Classify this image amongst the 1000 Imagenet classes: "{class_options}". Answer with only the class name.'
            # 'Say Hello and nothing else.'
            # f"Classify this image amongst the 1000 Imagenet classes. Answer with only the class name."
            # f"Answer with only the class name from these ImageNet 1K 1000 categories" # (no other word beyond these 1000 categories) : [{class_options}]."
        )

    def preprocess(self):
        """Return identity function as QWEN processor handles preprocessing"""
        return lambda x: x

    def _generate_response(self, pil_images, prompts, batch_size=16):
        """Generate responses from QWEN model"""
        all_responses = []

        with torch.no_grad():
            # Process images one by one to avoid batching issues
            for i, image in enumerate(
                tqdm(pil_images, desc="Generating Responses")
            ):
                prompt = prompts[i] if isinstance(prompts, list) else prompts

                # try:
                # Prepare conversation for QWEN format
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                # Apply chat template and tokenize
                inputs = self.processor.apply_chat_template(
                    [conversation],  # Wrap in list for single conversation
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                # Generate response with improved parameters
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=10,  # Short response - just want class name
                    min_new_tokens=1,  # Ensure at least one token
                    do_sample=True,  # Enable sampling for variety
                    temperature=0.1,  # Low temperature for focused responses
                    top_p=0.9,  # Nucleus sampling
                    repetition_penalty=1.2,  # Prevent repetition
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                )

                # Decode response
                response = self.processor.decode(
                    output[0], skip_special_tokens=True
                )

                # Extract only the assistant's response part
                # QWEN typically uses assistant tags or similar formatting
                if "<|im_start|>assistant" in response:
                    response = response.split("<|im_start|>assistant")[
                        -1
                    ].strip()
                    if "<|im_end|>" in response:
                        response = response.split("<|im_end|>")[0].strip()
                elif "assistant\n" in response:
                    response = response.split("assistant\n")[-1].strip()

                all_responses.append(response)

                # import pdb;pdb.set_trace()

                # except Exception as e:
                #     print(f"Error generating response for image {i}: {e}")
                #     all_responses.append("")

        return all_responses

    def _parse_response(self, response):
        """Parse the model response to extract the predicted class"""
        response = response.lower().strip()

        # Remove common prefixes/suffixes
        response = re.sub(r"^(the\s+)?(class\s+)?(is\s+)?", "", response)
        response = re.sub(r"\.$", "", response)

        # Try different regex patterns to extract the class name
        patterns = [
            r"'([^']+)'",  # Single quotes
            r'"([^"]+)"',  # Double quotes
            r"\b(\w+(?:\s+\w+)*)\b$",  # Last word(s) - handles multi-word classes
            r"([a-zA-Z_]+(?:\s+[a-zA-Z_]+)*)",  # Any word sequence with letters/underscores
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                candidate = match.group(1).strip()

                # Direct match check
                for class_name in self.class_names:
                    if class_name.lower() == candidate:
                        return class_name

                # Fuzzy match for underscores vs spaces
                for class_name in self.class_names:
                    if class_name.lower().replace(
                        " ", "_"
                    ) == candidate.replace(" ", "_"):
                        return class_name

                # Partial match check
                for class_name in self.class_names:
                    if (
                        candidate in class_name.lower()
                        or class_name.lower() in candidate
                    ):
                        return class_name

        # Fallback: check if any class name appears anywhere in the response
        for class_name in self.class_names:
            if class_name.lower() in response:
                return class_name

        # Final fallback: return the first class as default
        return self.class_names[0] if self.class_names else "unknown"

    def _convert_to_logits(self, predictions):
        """Convert string predictions to logits format"""
        batch_size = len(predictions)
        num_classes = len(self.class_names)

        # Create logits tensor
        logits = torch.zeros(batch_size, num_classes)

        for i, prediction in enumerate(predictions):
            # Find the index of the predicted class
            try:
                class_idx = self.class_names.index(prediction)
                # Set high confidence for predicted class, low for others
                logits[i, class_idx] = 10.0  # High confidence
                # Set small negative values for other classes
                for j in range(num_classes):
                    if j != class_idx:
                        logits[i, j] = -1.0
            except ValueError:
                # If prediction not in class_names, distribute equally (very low confidence)
                logits[i] = torch.ones(num_classes) * -5.0

        return logits

    def forward_batch(self, images):
        """Forward pass for batch of images"""
        assert isinstance(images, torch.Tensor)

        # Convert tensor images to PIL
        images = undo_default_preprocessing(images)
        pil_images = [ToPILImage()(image.cpu()) for image in images]

        self.model.eval()

        # Generate prompts for each image
        prompts = [self.question_template] * len(pil_images)

        # Get responses from the model
        responses = self._generate_response(
            pil_images, prompts, batch_size=self.batch_size
        )

        # Parse responses to get predicted classes
        predictions = [
            self._parse_response(response) for response in responses
        ]

        # Optional: Add debug breakpoint like in LLaVA version
        # import pdb; pdb.set_trace()

        # Convert predictions to logits format
        logits = self._convert_to_logits(predictions)

        return self.to_numpy(logits)

    def forward_single(self, image):
        """Process a single image"""
        return self.forward_batch(image.unsqueeze(0))[0]


import re

import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from tqdm import tqdm


class LlavaPytorchModel(PytorchModel):
    def __init__(self, model, processor, model_name, *args, **kwargs):
        super().__init__(model, model_name, *args)
        self.processor = processor
        self.device = device()

        # Extract batch_size from kwargs or set a default
        self.batch_size = kwargs.pop("batch_size", 32)

        # Store class names and templates for direct questioning
        self.class_names = kwargs.get("class_names", imagenet_classes)
        self.templates = kwargs.get("templates", imagenet_templates)

        # Create the direct question prompt
        self._setup_question_prompt()

    def _setup_question_prompt(self):
        """Setup the question prompt template for direct questioning"""
        # Create class options string
        class_options = "\n".join(self.class_names[:])

        # Create the base prompt template (without USER/ASSISTANT formatting)
        self.question_template = (
            f"You are an ImageNet classifier. Choose exactly one class label from the list provided by the user."
            f"Respond with *only* that class name—no quotes, no punctuation, no additional "
            f"words. If uncertain, guess the most likely class. Never apologise or mention "
            f"uncertainty. Classify this image among the 1000 ImageNet classes:\n{class_options}\n"
            # 'Classify this image amongst the 1000 Imagenet classes: "{class_options}". Answer with only the class name.'
            # 'Say Hello and nothing else.'
            # f"Classify this image amongst the 1000 Imagenet classes. Answer with only the class name."
            # f"Answer with only the class name from these ImageNet 1K 1000 categories" # (no other word beyond these 1000 categories) : [{class_options}]."
        )

    def preprocess(self):
        return lambda images: [
            (
                Image.fromarray(
                    (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                )
                if isinstance(img, torch.Tensor)
                else img
            )
            for img in images
        ]

    def _generate_response(self, pil_images, prompts, batch_size=16):
        """Generate responses from LLaVA model"""
        all_responses = []

        with torch.no_grad():
            # Process images one by one to avoid batching issues with LLaVA
            for i, image in enumerate(
                tqdm(pil_images, desc="Generating Responses")
            ):
                prompt = prompts[i] if isinstance(prompts, list) else prompts

                # try:
                # Format prompt for LLaVA-1.5 (USER/ASSISTANT format)
                formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

                # Prepare inputs - note the order: images first, then text
                inputs = self.processor(
                    images=image,
                    text=formatted_prompt,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                # Generate response with improved parameters
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=5,  # Very short - just want one word
                    min_new_tokens=1,  # Ensure at least one token
                    do_sample=True,  # Enable sampling for variety
                    temperature=0.1,  # Low temperature for focused responses
                    top_p=0.9,  # Nucleus sampling
                    repetition_penalty=2.0,  # Prevent repetition
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                )

                # Decode response
                response = self.processor.decode(
                    output[0], skip_special_tokens=True
                )

                # Extract only the assistant's response
                if "ASSISTANT:" in response:
                    response = response.split("ASSISTANT:")[-1].strip()

                all_responses.append(response)

                # except Exception as e:
                #     print(f"Error generating response for image {i}: {e}")
                #     all_responses.append("")

        return all_responses

    def _parse_response(self, response):
        """Parse the model response to extract the predicted class"""
        response = response.lower().strip()

        # Try different regex patterns to extract the class name
        patterns = [
            r"'([^']+)'",  # Single quotes
            r'"([^"]+)"',  # Double quotes
            r"\b(\w+)\b$",  # Last word
            r"([a-zA-Z_]+)",  # Any word with letters/underscores
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                candidate = match.group(1).strip()
                # Check if the candidate is one of our valid class names
                for class_name in self.class_names:
                    if class_name.lower().replace(
                        " ", "_"
                    ) == candidate.replace(" ", "_"):
                        return class_name
                    if class_name.lower() == candidate:
                        return class_name

        # Fallback: check if any class name appears in the response
        for class_name in self.class_names:
            if class_name.lower() in response:
                return class_name

        # If no match found, return the first class as default
        return self.class_names[0] if self.class_names else "unknown"

    def _convert_to_logits(self, predictions):
        """Convert string predictions to logits format"""
        batch_size = len(predictions)
        num_classes = len(self.class_names)

        # Create logits tensor
        logits = torch.zeros(batch_size, num_classes)

        for i, prediction in enumerate(predictions):
            # Find the index of the predicted class
            try:
                class_idx = self.class_names.index(prediction)
                # Set high confidence for predicted class, low for others
                logits[i, class_idx] = 10.0  # High confidence
                # Set small negative values for other classes
                for j in range(num_classes):
                    if j != class_idx:
                        logits[i, j] = -1.0
            except ValueError:
                # If prediction not in class_names, distribute equally
                logits[i] = torch.ones(num_classes) * 0.1

        return logits

    def forward_batch(self, images):
        assert isinstance(images, torch.Tensor)

        # Convert tensor images to PIL
        images = undo_default_preprocessing(images)
        pil_images = [ToPILImage()(image.cpu()) for image in images]

        self.model.eval()

        # Generate prompts for each image
        prompts = [self.question_template] * len(pil_images)

        # Get responses from the model
        responses = self._generate_response(
            pil_images, prompts, batch_size=self.batch_size
        )

        # Parse responses to get predicted classes
        predictions = [
            self._parse_response(response) for response in responses
        ]

        import pdb

        pdb.set_trace()
        # Convert predictions to logits format
        logits = self._convert_to_logits(predictions)

        return self.to_numpy(logits)

    def forward_single(self, image):
        """Process a single image"""
        return self.forward_batch(image.unsqueeze(0))[0]


# class LlavaPytorchModel(PytorchModel):
#     def __init__(self, model, processor, model_name, *args, **kwargs): # Add **kwargs here
#         super(LlavaPytorchModel, self).__init__(model, model_name, *args)
#         self.processor = processor
#         self.device = device()

#         # Extract batch_size from kwargs or set a default
#         self.batch_size = kwargs.pop('batch_size', 32) # Default to 32 if not provided

#         self.zeroshot_weights = self._get_zeroshot_weights(imagenet_classes, imagenet_templates, batch_size=self.batch_size)

#     def _get_zeroshot_weights(self, class_names, templates, batch_size): # Ensure batch_size is accepted here
#         all_class_embeddings = []
#         with torch.no_grad():
#             for class_name in tqdm(class_names, desc="Generating Zero-Shot Weights"):
#                 texts = [template.format(class_name) for template in templates]
#                 prompts_for_class = [f"USER: {text}\nASSISTANT:" for text in texts]

#                 text_embeddings_for_class = []
#                 for i in range(0, len(prompts_for_class), batch_size):
#                     batch_prompts = prompts_for_class[i:i + batch_size]

#                     inputs = self.processor(
#                         text=batch_prompts,
#                         return_tensors="pt",
#                         padding=True,
#                         truncation=True
#                     ).to(self.device)

#                     outputs = self.model.language_model(**inputs, output_hidden_states=True)
#                     batch_text_embeddings = outputs.last_hidden_state.mean(dim=1)
#                     text_embeddings_for_class.append(batch_text_embeddings)

#                 class_embedding = torch.cat(text_embeddings_for_class).mean(dim=0, keepdim=True)
#                 class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
#                 all_class_embeddings.append(class_embedding)

#             zeroshot_weights = torch.cat(all_class_embeddings, dim=0).to(self.device)
#         return zeroshot_weights

#     def preprocess(self):
#         return lambda images: [Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
#                                if isinstance(img, torch.Tensor) else img for img in images]


#     def _extract_image_features(self, pil_images, batch_size=16): # This function already has batch_size
#         """
#         Extract image features from LLaVA, projected into the language model's embedding space.
#         """
#         all_image_features = []
#         with torch.no_grad():
#             for i in tqdm(range(0, len(pil_images), batch_size), desc="Extracting Image Features"):
#                 batch_pil_images = pil_images[i:i + batch_size]
#                 dummy_text_prompts = [""] * len(batch_pil_images) # Simplest dummy text

#                 inputs = self.processor(
#                     text=dummy_text_prompts,
#                     images=batch_pil_images,
#                     return_tensors="pt",
#                     padding=True
#                 ).to(self.device)

#                 # Get pixel values, which are the input to the vision tower
#                 pixel_values = inputs["pixel_values"]

#                 # Pass pixel values through the vision tower
#                 vision_hidden_states = self.model.vision_tower(pixel_values).last_hidden_state

#                 # Pass the vision tower output through the multi-modal projector
#                 # This projects the vision features into the language model's embedding space
#                 image_features_projected = self.model.multi_modal_projector(vision_hidden_states)

#                 # Pool the projected image features
#                 # LLaVA typically has a <image> token in its prompt. The `image_features_projected`
#                 # usually corresponds to the embeddings that replace this token.
#                 # If `image_features_projected` is (batch_size, num_visual_tokens, embedding_dim),
#                 # we need to average over the visual tokens to get a single vector per image.
#                 # The 'num_visual_tokens' can vary, but for ViT-L/14, it's often 257 (256 patches + 1 CLS token).
#                 # After projection, these tokens are directly in the LM's embedding space.

#                 # Check dimensionality. If it's 3D (batch, tokens, dim), mean pool.
#                 if image_features_projected.dim() == 3:
#                     image_feature = image_features_projected.mean(dim=1)
#                 else: # Already pooled (batch, dim)
#                     image_feature = image_features_projected

#                 # Normalize the features
#                 image_feature /= image_feature.norm(dim=-1, keepdim=True)
#                 all_image_features.append(image_feature)

#         return torch.cat(all_image_features, dim=0)

#     def forward_batch(self, images):
#         assert isinstance(images, torch.Tensor)

#         images = undo_default_preprocessing(images)
#         pil_images = [ToPILImage()(image.cpu()) for image in images]

#         self.model.eval()

#         image_features = self._extract_image_features(pil_images, batch_size=self.batch_size) # Use self.batch_size here

#         # The dimensions should now match: image_features [batch_size, LLM_embedding_dim]
#         # and zeroshot_weights [num_classes, LLM_embedding_dim]
#         # Squeeze if there's an unnecessary dimension, but they should be [B, D] and [N, D]
#         # If your image_features or zeroshot_weights sometimes have an extra singleton dimension
#         # (e.g., [B, 1, D] or [N, 1, D]), then keep the squeeze logic.
#         # Otherwise, if they are always [B, D] and [N, D], you might not need to squeeze.
#         image_features = image_features.squeeze(1) if image_features.dim() == 3 else image_features
#         zeroshot_weights = self.zeroshot_weights.squeeze(1) if self.zeroshot_weights.dim() == 3 else self.zeroshot_weights

#         try:
#             logits = 100. * image_features @ zeroshot_weights.T
#         except Exception as e:
#             print(f"Error during matrix multiplication: {e}")
#             import pdb;pdb.set_trace()

#         return self.to_numpy(logits)


# --- Base VLM Model Class ---
class VLMModel(ABC):
    """
    Abstract base class for Vision-Language Models.
    Defines the common interface for VLM models.
    """

    def __init__(self, model, processor, model_name, *args, **kwargs):
        self.model = model
        self.processor = processor
        self.model_name = model_name
        self.device = device()
        self.model.to(self.device)
        self.zeroshot_weights = None  # Will be computed by concrete classes
        # Determine the model's preferred dtype for consistency (e.g., float16)
        self.target_dtype = (
            self.model.dtype if hasattr(self.model, "dtype") else torch.float16
        )

    @abstractmethod
    def _get_zeroshot_weights(self, class_names, templates):
        """
        Abstract method to generate zero-shot classification weights.
        Implementations will vary based on the model's text encoder.
        """

    @abstractmethod
    def _extract_image_features(self, pil_images):
        """
        Abstract method to extract image features.
        Implementations will vary based on the model's vision encoder.
        """

    def preprocess(self):
        """
        Returns a preprocessing function.
        For models using Hugging Face processors, this often returns an identity function
        as the processor handles everything.
        """
        return lambda x: x  # Default: processor handles preprocessing

    def to_numpy(self, tensor):
        """Converts a torch tensor to a numpy array."""
        return tensor.detach().cpu().numpy()

    def softmax(self, logits):
        """
        Applies softmax to the logits.
        Expects logits as a numpy array and returns probabilities as a numpy array.
        """
        logits_tensor = torch.from_numpy(logits).to(self.device)
        probabilities_tensor = softmax(logits_tensor, dim=-1)
        return self.to_numpy(probabilities_tensor)

    def forward_batch(self, images):
        """
        Performs a forward pass for a batch of images for zero-shot classification.
        Assumes images are already in the expected format (e.g., PyTorch tensors).
        """
        if self.zeroshot_weights is None:
            raise ValueError(
                "Zero-shot weights have not been computed. Call _get_zeroshot_weights first."
            )

        # Ensure images are PIL format for feature extraction
        pil_images = [ToPILImage()(image.cpu()) for image in images]

        self.model.eval()
        with torch.no_grad():
            # Extract image features
            image_features = self._extract_image_features(pil_images)

            # Ensure dtype alignment before matrix multiplication
            # Option 1: Convert both to model's preferred dtype (often float16 for speed)
            target_dtype = (
                self.model.dtype
            )  # Get the dtype the model was loaded with
            image_features = image_features.to(target_dtype)
            zeroshot_weights_T = self.zeroshot_weights.T.to(target_dtype)

            # Option 2 (less common if model is float16): Convert both to float32 for maximum precision
            # image_features = image_features.to(torch.float32)
            # zeroshot_weights_T = self.zeroshot_weights.T.to(torch.float32)

            # Compute similarity with zero-shot weights
            logits = 100.0 * image_features @ zeroshot_weights_T

        return self.to_numpy(logits)


# --- BLIP-2 Model Implementation ---
class Blip2PytorchModel(VLMModel):
    def __init__(self, model, processor, model_name, *args, **kwargs):
        super().__init__(model, processor, model_name, *args, **kwargs)
        # Ensure zero-shot weights are initialized here
        self.zeroshot_weights = self._get_zeroshot_weights(
            imagenet_classes, imagenet_templates
        )

    def _get_zeroshot_weights(self, class_names, templates, batch_size=32):
        """
        Generate zero-shot classification weights using BLIP-2's text capabilities.
        This adapts the Qwen-like approach for BLIP-2, treating text prompts
        as "conversations" to get language model hidden states.
        """
        print(
            "Generating Zero-Shot Weights for BLIP-2 using adapted text encoding."
        )
        all_class_embeddings = []
        target_dtype = self.model.dtype

        with torch.no_grad():
            for class_name in tqdm(
                class_names, desc="Generating BLIP-2 Zero-Shot Weights"
            ):
                texts = [template.format(class_name) for template in templates]
                text_embeddings_for_class = []

                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]

                    # BLIP-2's processor for text only
                    text_inputs = self.processor(
                        text=batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(self.device)

                    # For BLIP-2's language model, a dummy pixel_values is usually required
                    dummy_pixel_values = torch.zeros(
                        len(batch_texts),
                        3,
                        self.processor.image_processor.size["height"],
                        self.processor.image_processor.size["width"],
                        dtype=target_dtype,  # Ensure dummy pixel values match model dtype
                    ).to(self.device)

                    outputs = self.model(
                        pixel_values=dummy_pixel_values,
                        input_ids=text_inputs.input_ids,
                        attention_mask=text_inputs.attention_mask,
                        output_hidden_states=True,  # Crucial for getting embeddings
                    )

                    # Corrected: Access the last hidden state from the 'hidden_states' tuple
                    if (
                        hasattr(outputs, "language_model_outputs")
                        and outputs.language_model_outputs is not None
                    ):
                        # Access the last hidden state from the hidden_states tuple
                        lm_last_hidden_state = (
                            outputs.language_model_outputs.hidden_states[-1]
                        )
                    else:
                        raise AttributeError(
                            "Could not find language_model_outputs or its hidden_states. Ensure output_hidden_states=True and model returns them correctly."
                        )

                    # Mean pooling over tokens, masked by attention_mask
                    attention_mask_expanded = (
                        text_inputs.attention_mask.unsqueeze(-1)
                        .expand(lm_last_hidden_state.size())
                        .float()
                    )
                    # Ensure mask also matches target_dtype
                    attention_mask_expanded = attention_mask_expanded.to(
                        target_dtype
                    )

                    masked_embeddings = (
                        lm_last_hidden_state * attention_mask_expanded
                    )
                    sum_embeddings = torch.sum(masked_embeddings, 1)
                    sum_mask = torch.clamp(
                        attention_mask_expanded.sum(1), min=1e-9
                    )

                    text_embeddings_for_class.append(sum_embeddings / sum_mask)

                class_embedding = torch.cat(text_embeddings_for_class).mean(
                    dim=0, keepdim=True
                )
                class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
                all_class_embeddings.append(
                    class_embedding.to(target_dtype)
                )  # Ensure final weights match target_dtype

            zeroshot_weights = torch.cat(all_class_embeddings, dim=0).to(
                self.device
            )
        return zeroshot_weights

    def _extract_image_features(self, pil_images, batch_size=16):
        """
        Extract image features using BLIP-2's vision encoder by processing images in batches.
        """
        all_image_features = []
        target_dtype = self.model.dtype  # Get the model's dtype
        with torch.no_grad():
            for i in tqdm(
                range(0, len(pil_images), batch_size),
                desc="Extracting BLIP-2 Image Features",
            ):
                batch_pil_images = pil_images[i : i + batch_size]

                # BLIP-2 processor handles image preprocessing
                inputs = self.processor(
                    images=batch_pil_images, return_tensors="pt"
                ).to(self.device)

                # Ensure pixel_values dtype matches model's dtype
                pixel_values = inputs["pixel_values"].to(target_dtype)

                # Directly get image features from the model
                image_features = self.model.get_image_features(
                    pixel_values=pixel_values
                )

                # BLIP-2's image features are often 3D (batch, sequence_length, hidden_size).
                # Mean pool over sequence length if needed, then normalize.
                if image_features.dim() == 3:
                    image_features = image_features.mean(dim=1)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                all_image_features.append(image_features)

        return torch.cat(all_image_features, dim=0).to(
            target_dtype
        )  # Ensure output matches target_dtype
