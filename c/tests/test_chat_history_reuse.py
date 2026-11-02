"""A chat's next turn must begin with what the engine already holds.

After a turn the engine's state covers the prompt and every token it generated.
The client then resends the transcript with the answer as an assistant message
and, from colibri's web, the reasoning as `reasoning_content`. When the rendered
history reproduces the turn as it was generated, the next prompt begins with the
state and the engine reads only the new text. When it does not, an engine whose
reuse is all or nothing (a recurrent or windowed state: Qwen3.6, DeepSeek V4.1,
Inkling) reads the whole conversation again, every turn.

Each family's generated turn is written here by hand, from the format its model
emits after the generation cue, not derived from the history renderer, so the
test pins one against the other. Kimi K3 is not here: its prompt is assembled in
kimi_k3.c from length-framed records, not rendered as text.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import openai_server as o  # noqa: E402

QUESTION = "Explain how a hash map works."
ANSWER = "It stores pairs in buckets chosen by a hash of the key."
REASONING = "The user wants the idea and the collision handling."
FOLLOW_UP = "Now an example in C."


def qwen36(messages, thinking):
    return o.render_chat_qwen(messages, thinking, None, None, None, True,
                              o.preserve_thinking_default(messages, thinking))


# family: (renderer, the turn the model generates with thinking off, with thinking on)
FAMILIES = {
    "qwen36": (qwen36, ANSWER, f"{REASONING}\n</think>\n\n{ANSWER}"),
    "qwen38": (lambda m, t: o.render_chat_qwen38(m, t, None, None, None, True),
               ANSWER, f"{REASONING}\n</think>\n\n{ANSWER}"),
    "deepseek_v4": (lambda m, t: o.render_chat_v4(m, t, None, None, None, True),
                    ANSWER, f"{REASONING}</think>{ANSWER}"),
    "deepseek_v41": (lambda m, t: o.render_chat_dsv41(m, t, None, None, None, True),
                     ANSWER, f"{REASONING}</think>{ANSWER}"),
    "inkling": (lambda m, t: o.render_chat_inkling(m, t, None, None, None, add_generation_prompt=True),
                ANSWER,
                f"<|content_thinking|>{REASONING}<|end_message|><|message_model|><|content_text|>{ANSWER}"),
    "mimo": (lambda m, t: o.render_chat_mimo(m, t, None, None, None, True),
             ANSWER, f"{REASONING}</think>{ANSWER}"),
    "glm53": (lambda m, t: o.render_chat_glm53(m, t, None, None, None, True),
              f"</think>{ANSWER}", f"{REASONING}</think>{ANSWER}"),
    "glm": (lambda m, t: o.render_chat(m, t, None, None, None, True),
            ANSWER, f"{REASONING}</think>{ANSWER}"),
    "olmoe": (lambda m, t: o.render_chat_olmoe(m, t, None, None, None, True), ANSWER, None),
}


class NextTurnExtendsTheState(unittest.TestCase):
    def check(self, family, thinking):
        render, generated_off, generated_on = FAMILIES[family]
        generated = generated_on if thinking else generated_off
        if generated is None:
            self.skipTest(f"{family} has no thinking mode")
        first = render([{"role": "user", "content": QUESTION}], thinking)
        reply = {"role": "assistant", "content": ANSWER}
        if thinking:
            reply["reasoning_content"] = REASONING
        second = render([{"role": "user", "content": QUESTION}, reply,
                         {"role": "user", "content": FOLLOW_UP}], thinking)
        held = first + generated
        if not second.startswith(held):
            at = next((i for i, (a, b) in enumerate(zip(held, second)) if a != b),
                      min(len(held), len(second)))
            self.fail(f"{family}, thinking {'on' if thinking else 'off'}: the next turn's prompt "
                      f"leaves the engine's state at character {at} of {len(held)}: it holds "
                      f"{held[at:at + 60]!r}, the history renders {second[at:at + 60]!r}")

    def test_every_family_thinking_off(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                self.check(family, False)

    def test_every_family_thinking_on_with_the_reasoning_sent_back(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                self.check(family, True)

    def test_without_reasoning_sent_back_the_templates_keep_their_default(self):
        """A standard client sends no reasoning: the families that strip past reasoning
        by default still do (Qwen3.6 with thinking on renders no block; DeepSeek V4.1
        renders the bare </think>), so nothing changes for it."""
        reply = {"role": "assistant", "content": ANSWER}
        history = [{"role": "user", "content": QUESTION}, reply,
                   {"role": "user", "content": FOLLOW_UP}]
        self.assertIn(f"<|im_start|>assistant\n{ANSWER}<|im_end|>", qwen36(history, True))
        self.assertIn(f"</think>{ANSWER}", o.render_chat_dsv41(history, True, None, None, None, True))
        self.assertNotIn("<|content_thinking|>",
                         o.render_chat_inkling(history, True, None, None, None, add_generation_prompt=True))

    def test_inkling_renders_reasoning_as_its_template_does(self):
        """chat_template.jinja: the reasoning is a model message of its own, before the text."""
        history = [{"role": "user", "content": QUESTION},
                   {"role": "assistant", "content": ANSWER, "reasoning_content": REASONING},
                   {"role": "user", "content": FOLLOW_UP}]
        rendered = o.render_chat_inkling(history, False, None, None, None, add_generation_prompt=True)
        self.assertIn(f"<|message_model|><|content_thinking|>{REASONING}<|end_message|>"
                      f"<|message_model|><|content_text|>{ANSWER}<|end_message|>"
                      "<|content_model_end_sampling|>", rendered)
        bad = [{"role": "user", "content": QUESTION},
               {"role": "assistant", "content": ANSWER, "reasoning_content": 5}]
        with self.assertRaises(o.APIError):
            o.render_chat_inkling(bad, False, None, None, None, add_generation_prompt=True)


if __name__ == "__main__":
    unittest.main()
