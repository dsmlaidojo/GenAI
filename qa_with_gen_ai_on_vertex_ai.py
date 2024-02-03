#! /bin/env python3

import pandas as pd
from vertexai.language_models import TextGenerationModel

generation_model = TextGenerationModel.from_pretrained("text-bison@001")

prompt = "Tell me top five reasons why OpenAI is better than Vertex AI."
print(
    generation_model.predict(
        prompt,
        max_output_tokens=256,
        temperature=0.1,
    ).text
)


