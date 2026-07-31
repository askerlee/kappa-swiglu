"""
Base class for all Tasks.
A Task is basically a dataset of conversations, together with some
metadata and often also evaluation criteria.
Example tasks: MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval, SmolTalk.
"""

import random

class Task:
    """
    Base class of a Task. Allows for lightweight slicing of the underlying dataset.
    """

    def __init__(self, start=0, stop=None, step=1):
        # allows a lightweight logical view over a dataset
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be greater than or equal to start, got {stop} and {start}"
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop # could be None here
        self.step = step

    @property
    def eval_type(self):
        # one of 'generative' | 'categorical'
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}" # prevent footguns
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index)
        return conversation

    def evaluate(self, problem, completion):
        raise NotImplementedError


class TaskMixture(Task):
    """
    For SFT Training it becomes useful to train on a mixture of datasets.
    Fun trick: if you wish to oversample any task, just pass it in multiple times in the list.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        # tasks is a list of Task objects
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        # Build list of all (task_idx, local_idx) pairs
        self.index_map = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))
        # Deterministically shuffle to mix tasks throughout training
        rng = random.Random(42)
        rng.shuffle(self.index_map)
        # Note: this is not the most elegant or best solution, but it's ok for now

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        """
        Access conversations according to a deterministic shuffle of all examples.
        This ensures tasks are mixed throughout training, regardless of dataset size.
        """
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for mixture with {self.num_conversations} conversations"
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class TaskSequence(Task):
    """
    For SFT Training sometimes we want to sequentially train on a list of tasks.
    This is useful for cases that require a training curriculum.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for sequence with {self.num_conversations} conversations"
        for task_idx, task_length in enumerate(self.lengths):
            if index < task_length:
                return self.tasks[task_idx][index]
            index -= task_length


def normalize_conversation(messages):
    """
    Normalize a list of messages into the strict user/assistant alternating form
    expected by render_conversation.

    Rules:
    1. An optional leading system message is kept as-is (render_conversation handles it).
    2. After (optionally) stripping the system message, messages must strictly alternate
       starting with "user", e.g. user, assistant, user, assistant, ...
    3. If two consecutive messages share the same role, they are merged into one.
       (e.g. two consecutive assistant turns become a single assistant message.)
    4. String content is concatenated; lists of parts are concatenated; mixed types
       are coerced to a list of parts.
    5. Messages with empty content after merging are dropped (unless that leaves no
       messages at all).

    Returns the (possibly rewritten) list of messages.
    """
    messages = list(messages)

    # Peel off an optional leading system message (keep it for render_conversation to handle).
    system_message = None
    if messages and messages[0]["role"] == "system":
        system_message = messages[0]
        rest = messages[1:]
    else:
        rest = messages

    if not rest:
        # Degenerate: only a system message (or nothing). Return as-is so downstream asserts fire.
        return messages

    # Coerce every message content to a comparable form: list of parts.
    def to_parts(content):
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        if isinstance(content, list):
            return list(content)
        if content is None:
            return []
        return [{"type": "text", "text": str(content)}]

    # Merge consecutive messages that share the same role.
    merged = []
    for msg in rest:
        role = msg["role"]
        parts = to_parts(msg["content"])
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"].extend(parts)
        else:
            merged.append({"role": role, "content": parts})

    # Drop messages whose content became empty after merging, then re-merge in case of gaps.
    merged = [msg for msg in merged if len(msg["content"]) > 0]

    # Re-merge after dropping empties (could create new adjacent same-role pairs).
    final = []
    for msg in merged:
        if final and final[-1]["role"] == msg["role"]:
            final[-1]["content"].extend(msg["content"])
        else:
            final.append({"role": msg["role"], "content": list(msg["content"])})

    # Re-coerce: single text-part messages become plain strings for cleanliness.
    for msg in final:
        parts = msg["content"]
        if msg["role"] == "user":
            # User messages must be plain strings (render_conversation asserts this).
            # Concatenate all text parts into a single string.
            text_parts = [p.get("text", "") for p in parts if p.get("type") == "text"]
            msg["content"] = "\n\n".join(t for t in text_parts if t)
            if not msg["content"]:
                msg["content"] = "..."
        elif len(parts) == 1 and parts[0].get("type") == "text":
            msg["content"] = parts[0]["text"]
        else:
            msg["content"] = parts

    # Ensure alternation starting with user. If the first role is assistant, prepend a
    # minimal user turn so the conversation is well-formed.
    if final and final[0]["role"] != "user":
        final.insert(0, {"role": "user", "content": "..."})

    # If we have an odd number of messages (e.g. trailing user with no reply), drop the last one.
    if len(final) % 2 == 1 and final and final[-1]["role"] == "user":
        final = final[:-1]

    if not final:
        # Nothing survived normalization; return original so downstream error is clearer.
        return messages

    result = ([system_message] if system_message else []) + final
    return result


def render_mc(question, letters, choices):
    """
    The common multiple choice rendering format we will use.

    Note two important design decisions:
    1)
    Bigger models don't care as much, but smaller models prefer to have
    the letter *after* the choice, which results in better binding.
    2)
    There is no whitespace between the delimiter (=) and the letter.
    This is actually critical because the tokenizer has different token ids
    for " A" vs. "A". The assistant responses will be just the letter itself,
    i.e. "A", so it is important that here in the prompt it is the exact same
    token, i.e. "A" with no whitespace before it. Again, bigger models don't care
    about this too much, but smaller models do care about some of these details.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query


if __name__ == "__main__":
    # very lightweight test of slicing
    from tasks.mmlu import MMLU

    ds = MMLU(subset="auxiliary_train", split="train")
    print("Length of MMLU: ", len(ds))
    ex = ds[5]
    print("5th example: ", ex)

    ds = MMLU(subset="auxiliary_train", split="train", start=5, stop=10)
    print("Length of sliced MMLU[5:10]: ", len(ds))
    print("0th example of sliced MMLU: ", ds[0])

    print("They match: ", ex == ds[0])
