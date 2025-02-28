# model_repository/pth/1/model.py
import triton_python_backend_utils as pb_utils
import numpy as np
import transformers
import Path
import torch


class TritonPythonModel:
    def initialize(self, args):
        self.model_path: str = str(Path(args["model_repository"]).parent.absolute())

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)
        model_config = transformers.AutoConfig.from_pretrained(self.model_path)
        self.model = transformers.AutoModelForSequenceClassification(
            config=model_config, device=self.device
        )
        self.model.eval()

    def execute(self, requests):
        responses = []
        for request in requests:
            input_ids = pb_utils.get_input_tensor_by_name(
                request, "input_ids"
            ).as_numpy()
            attention_mask = pb_utils.get_input_tensor_by_name(
                request, "attention_mask"
            ).as_numpy()

            input_ids = torch.from_numpy(input_ids).long().to(self.device)
            attention_mask = torch.from_numpy(attention_mask).long().to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits.cpu().numpy()

            output_tensor = pb_utils.Tensor("output", logits.astype(np.float32))
            response = pb_utils.PBOutput(output_tensors=[output_tensor])
            responses.append(response)
        return responses

    def finalize(self):
        pass
