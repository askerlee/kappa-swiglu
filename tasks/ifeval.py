"""IFEval strict instruction-following evaluation."""

import json
import re

from datasets import load_dataset

from tasks.common import Task


def _count_words(text):
    return len(re.findall(r"\w+", text))


def _count_sentences(text):
    return len([sentence for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence])


def _relation_passes(actual, threshold, relation):
    if relation == "less than":
        return actual < threshold
    if relation == "at least":
        return actual >= threshold
    raise ValueError(f"Unsupported IFEval relation: {relation}")


def _check_instruction(instruction_id, kwargs, prompt, response):
    """Evaluate one IFEval instruction using its official strict semantics."""
    if instruction_id == "keywords:existence":
        return all(re.search(keyword, response, flags=re.IGNORECASE) for keyword in kwargs["keywords"])
    if instruction_id == "keywords:frequency":
        return _relation_passes(
            len(re.findall(kwargs["keyword"], response, flags=re.IGNORECASE)),
            kwargs["frequency"],
            kwargs["relation"],
        )
    if instruction_id == "keywords:forbidden_words":
        return not any(re.search(r"\b" + word + r"\b", response, flags=re.IGNORECASE) for word in kwargs["forbidden_words"])
    if instruction_id == "keywords:letter_frequency":
        return _relation_passes(response.lower().count(kwargs["letter"].lower()), kwargs["let_frequency"], kwargs["let_relation"])
    if instruction_id == "length_constraints:number_sentences":
        return _relation_passes(_count_sentences(response), kwargs["num_sentences"], kwargs["relation"])
    if instruction_id == "length_constraints:number_paragraphs":
        paragraphs = re.split(r"\s?\*\*\*\s?", response)
        count = len(paragraphs) - int(not paragraphs[0].strip()) - int(not paragraphs[-1].strip())
        return count == kwargs["num_paragraphs"] and all(paragraph.strip() for paragraph in paragraphs[1:-1])
    if instruction_id == "length_constraints:number_words":
        return _relation_passes(_count_words(response), kwargs["num_words"], kwargs["relation"])
    if instruction_id == "length_constraints:nth_paragraph_first_word":
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\n", response) if paragraph.strip()]
        index = kwargs["nth_paragraph"] - 1
        if len(paragraphs) != kwargs["num_paragraphs"] or index >= len(paragraphs):
            return False
        first_word = re.match(r"[A-Za-z]+", paragraphs[index])
        return first_word is not None and first_word.group().lower() == kwargs["first_word"].lower()
    if instruction_id == "detectable_content:number_placeholders":
        return len(re.findall(r"\[.*?\]", response)) >= kwargs["num_placeholders"]
    if instruction_id == "detectable_content:postscript":
        marker = kwargs["postscript_marker"].lower().replace(".", r"\.?")
        return bool(re.search(r"\s*" + marker + r".*$", response.lower(), flags=re.MULTILINE))
    if instruction_id == "detectable_format:number_bullet_lists":
        bullets = re.findall(r"^\s*(?:\*[^*]|-).*?$", response, flags=re.MULTILINE)
        return len(bullets) == kwargs["num_bullets"]
    if instruction_id == "detectable_format:constrained_response":
        return any(option in response.strip() for option in ("My answer is yes.", "My answer is no.", "My answer is maybe."))
    if instruction_id == "detectable_format:number_highlighted_sections":
        highlights = re.findall(r"(?<!\*)\*[^\n*]+\*(?!\*)", response)
        highlights += re.findall(r"\*\*[^\n*]+\*\*", response)
        return len([highlight for highlight in highlights if highlight.strip("*").strip()]) >= kwargs["num_highlights"]
    if instruction_id == "detectable_format:multiple_sections":
        pattern = r"\s?" + re.escape(kwargs["section_spliter"]) + r"\s?\d+\s?"
        return len(re.split(pattern, response)) - 1 >= kwargs["num_sections"]
    if instruction_id == "detectable_format:json_format":
        content = response.strip()
        for prefix in ("```json", "```Json", "```JSON", "```"):
            if content.startswith(prefix):
                content = content.removeprefix(prefix).removesuffix("```").strip()
                break
        try:
            json.loads(content)
        except ValueError:
            return False
        return True
    if instruction_id == "detectable_format:title":
        return bool(re.search(r"<<[^\n<>]+>>", response))
    if instruction_id == "multi-turn:constrained_start":
        return bool(re.search(r"^\s*" + re.escape(kwargs["starter"]) + r".*$", response, flags=re.MULTILINE))
    if instruction_id == "combination:two_responses":
        responses = [part.strip() for part in response.split("******") if part.strip()]
        return len(responses) == 2 and responses[0] != responses[1]
    if instruction_id == "combination:repeat_prompt":
        return response.strip().lower().startswith(kwargs["prompt_to_repeat"].strip().lower())
    if instruction_id == "startend:end_checker":
        return response.strip().strip('"').lower().endswith(kwargs["end_phrase"].strip().lower())
    if instruction_id == "change_case:capital_word_frequency":
        return _relation_passes(len([word for word in re.findall(r"\b\w+(?:-\w+)*\b", response) if word.isupper()]), kwargs["capital_frequency"], kwargs["capital_relation"])
    if instruction_id == "change_case:english_capital":
        return response.isupper()
    if instruction_id == "change_case:english_lowercase":
        return response.islower()
    if instruction_id == "punctuation:no_comma":
        return "," not in response
    if instruction_id == "startend:quotation":
        stripped = response.strip()
        return len(stripped) > 1 and stripped.startswith('"') and stripped.endswith('"')
    if instruction_id == "language:response_language":
        try:
            from langdetect import detect
            return detect(response) == kwargs["language"]
        except ImportError as error:
            raise RuntimeError("IFEval language checks require the langdetect package.") from error
    raise ValueError(f"Unsupported IFEval instruction: {instruction_id}")


class IFEval(Task):
    """The 541-prompt strict instruction-following benchmark."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("google/IFEval", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        return {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": ""},
            ],
            "instruction_id_list": row["instruction_id_list"],
            "kwargs": row["kwargs"],
        }

    def evaluate(self, conversation, completion):
        prompt = conversation["messages"][0]["content"]
        return bool(completion.strip()) and all(
            _check_instruction(instruction_id, kwargs, prompt, completion)
            for instruction_id, kwargs in zip(conversation["instruction_id_list"], conversation["kwargs"])
        )