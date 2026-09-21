import logging

import numpy as np
import sentencepiece

import openpi.shared.download as download


class PaligemmaTokenizer:
    """Tokenizer for the pi05 prompt/state prefix."""

    def __init__(self, max_len: int = 200):
        self._max_len = max_len
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is None:
            full_prompt = f"{cleaned_text}\n"
        else:
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
        tokens = self._tokenizer.encode(full_prompt, add_bos=True)

        if len(tokens) > self._max_len:
            logging.warning(
                "Token length (%s) exceeds max length (%s), truncating.", len(tokens), self._max_len
            )
        tokens = tokens[: self._max_len]
        mask = [True] * len(tokens)
        if len(tokens) < self._max_len:
            padding = self._max_len - len(tokens)
            tokens += [False] * padding
            mask += [False] * padding
        return np.asarray(tokens), np.asarray(mask)
