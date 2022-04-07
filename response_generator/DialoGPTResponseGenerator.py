from transformers import AutoModelForCausalLM, AutoTokenizer


class DialoGPTResponseGenerator:
    def __init__(self, size="large"):
        self.tokenizer = AutoTokenizer.from_pretrained(f"microsoft/DialoGPT-{size}")
        self.model = AutoModelForCausalLM.from_pretrained(
            f"microsoft/DialoGPT-{size}"
        ).cuda()

    def __call__(self, context):
        history = (
            self.tokenizer.eos_token.join(context.get_text_history())
            + self.tokenizer.eos_token
        )

        gpt_input = self.tokenizer.encode(history, return_tensors="pt").cuda()

        chat_history_ids = self.model.generate(
            gpt_input,
            max_length=1000,
            temperature=0.6,
            repetition_penalty=1.3,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(
            chat_history_ids[:, gpt_input.shape[-1] :][0], skip_special_tokens=True
        )

        context.add_response_text(response)

        return response
