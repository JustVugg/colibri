"""tool_choice reaches every renderer: a forced function is offered alone and named, "none" offers no tools."""

import unittest

import openai_server as srv

TOOLS = [
    {"type": "function", "function": {"name": "search", "description": "Search the web.",
                                       "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "weather", "description": "Get the weather.",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}},
]
MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Rome?"}]
FORCED = {"type": "function", "function": {"name": "weather"}}
NAMED = "You must call the function `weather`"


def render(renderer, tool_choice):
    return renderer([dict(m) for m in MESSAGES], tools=[dict(t) for t in TOOLS], tool_choice=tool_choice)


class ForcedFunction(unittest.TestCase):
    RENDERERS = ("render_chat_qwen", "render_chat_qwen38", "render_chat_mimo", "render_chat_glm53",
                 "render_chat_v4")

    def test_a_forced_function_is_offered_alone_and_named(self):
        for name in self.RENDERERS:
            with self.subTest(renderer=name):
                prompt = render(getattr(srv, name), FORCED)
                self.assertIn('"weather"', prompt)
                self.assertNotIn('"search"', prompt)
                self.assertIn(NAMED, prompt)

    def test_auto_and_required_are_unchanged(self):
        for name in self.RENDERERS:
            with self.subTest(renderer=name):
                for choice in (None, "auto", "required"):
                    prompt = render(getattr(srv, name), choice)
                    self.assertIn('"search"', prompt)
                    self.assertIn('"weather"', prompt)
                    self.assertNotIn("You must call the function `", prompt)


class MistralLarge4ToolChoice(unittest.TestCase):
    def test_none_offers_no_tools(self):
        prompt = render(srv.render_chat_mistral_large4, "none")
        self.assertNotIn("[AVAILABLE_TOOLS]", prompt)

    def test_a_forced_function_is_offered_alone(self):
        prompt = render(srv.render_chat_mistral_large4, FORCED)
        self.assertIn('"weather"', prompt)
        self.assertNotIn('"search"', prompt)

    def test_auto_and_required_offer_every_tool(self):
        for choice in (None, "auto", "required"):
            with self.subTest(tool_choice=choice):
                prompt = render(srv.render_chat_mistral_large4, choice)
                self.assertIn('"search"', prompt)
                self.assertIn('"weather"', prompt)


if __name__ == "__main__":
    unittest.main()
