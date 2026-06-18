from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


SPECIAL_TOKENS = ["<bot>", "<eot>", "<latent>"]


class CoconutLM(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        trust_remote_code: bool = True,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        added_tokens = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS}
        )

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )

        if added_tokens > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.bot_id = self.tokenizer.convert_tokens_to_ids("<bot>")
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<eot>")
        self.latent_id = self.tokenizer.convert_tokens_to_ids("<latent>")

    @property
    def dev(self) -> torch.device:
        return next(self.parameters()).device

    def tokenize(self, text: str) -> dict:
        return self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )

    def make_latent_inputs_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Replaces each <latent> token embedding with the previous hidden state.

        Important:
        This function avoids in-place tensor updates because in-place updates
        break PyTorch autograd during backpropagation.
        """
        if input_ids.shape[0] != 1:
            raise ValueError(
                "This simple Coconut implementation supports batch_size=1 only."
            )

        embed_layer = self.model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)

        latent_positions = (
            input_ids[0] == self.latent_id
        ).nonzero(as_tuple=True)[0].tolist()

        for pos in latent_positions:
            prefix_embeds = inputs_embeds[:, :pos, :]

            attention_mask = torch.ones(
                prefix_embeds.shape[:2],
                device=prefix_embeds.device,
                dtype=torch.long,
            )

            outputs = self.model(
                inputs_embeds=prefix_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            previous_hidden = outputs.hidden_states[-1][:, -1:, :]

            # Avoid this in-place update:
            # inputs_embeds[:, pos, :] = previous_hidden
            #
            # Instead, rebuild the tensor safely using torch.cat().
            inputs_embeds = torch.cat(
                [
                    inputs_embeds[:, :pos, :],
                    previous_hidden,
                    inputs_embeds[:, pos + 1 :, :],
                ],
                dim=1,
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        input_ids = input_ids.to(self.dev)

        if labels is not None:
            labels = labels.to(self.dev)

        inputs_embeds = self.make_latent_inputs_embeds(input_ids)

        attention_mask = torch.ones(
            input_ids.shape,
            device=self.dev,
            dtype=torch.long,
        )

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

        return outputs

    @torch.no_grad()
    def generate_with_latents(
        self,
        prompt: str,
        k_latents: int = 2,
        max_new_tokens: int = 80,
    ) -> str:
        self.eval()

        seed_text = prompt + " <bot>" + (" <latent>" * k_latents) + " <eot>"

        input_ids = self.tokenize(seed_text)["input_ids"].to(self.dev)

        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids)

            next_token_id = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            input_ids = torch.cat(
                [input_ids, next_token_id],
                dim=1,
            )

            if self.tokenizer.eos_token_id is not None:
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        generated_text = self.tokenizer.decode(
            input_ids[0],
            skip_special_tokens=False,
        )

        return generated_text