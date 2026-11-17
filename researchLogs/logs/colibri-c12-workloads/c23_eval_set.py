"""C23 eval set — the committed, fixed question set for quant-quality scoring.

Everything a container sees is literal here (plus c23_eval_doc.txt): the same
bytes go to every arm, so cross-container differences are the model, not the
harness. Categories and scoring:

  code    (7)  generate a named Python function; the harness appends the task's
               asserts to the model's fenced block and RUNS it (subprocess,
               timeout). Score = 1 iff every assert passes. This is the graded
               form of the q2 Fibonacci probe that separated IQ2_XXS from
               IQ3/Q4_K_M in C16/C21.
  math    (6)  fresh multi-step word problems; answer extracted from a final
               'Answer: <integer>' line (fallback: last standalone integer).
               Exact match. Answers computed programmatically at authoring time.
  format  (5)  strict mechanical checks: JSON schema/sort, bullet count and
               length cap, exact tool-call schema, exact computed hex, exact
               sorted string.
  recall  (3)  required-keyword scoring; the weak scorer, flagged as anecdote.
  longctx (1)  the committed ledger document (~4.1K tokens) + 4 aggregation /
               needle questions asked over one conversation (append turns), so
               the block costs one prefill plus four tails. Answers computed at
               authoring time from the generator that produced the document.
"""

CODE_TASKS = [
    ("fib", 512,
     "Write a Python function named fib_iter(n) that returns the n-th Fibonacci "
     "number iteratively, with fib_iter(0) == 0 and fib_iter(1) == 1. Respond with "
     "a single fenced ```python code block containing only the function definition "
     "— no example usage, no explanation.",
     """
assert fib_iter(0) == 0
assert fib_iter(1) == 1
assert fib_iter(2) == 1
assert fib_iter(10) == 55
assert fib_iter(30) == 832040
print("ALL_OK")
"""),
    ("brackets", 512,
     "Write a Python function named is_balanced(s) that returns True if the string "
     "s contains only correctly nested and matched (), [] and {} brackets (an empty "
     "string is balanced), else False. Respond with a single fenced ```python code "
     "block containing only the function definition — no example usage, no explanation.",
     """
assert is_balanced("([]{})") is True
assert is_balanced("([)]") is False
assert is_balanced("") is True
assert is_balanced("(((") is False
assert is_balanced("{[()]}") is True
assert is_balanced(")(") is False
print("ALL_OK")
"""),
    ("roman", 512,
     "Write a Python function named roman_to_int(s) that converts a Roman numeral "
     "string (I, V, X, L, C, D, M, using subtractive notation where a smaller value "
     "precedes a larger one) to its integer value. Respond with a single fenced "
     "```python code block containing only the function definition — no example "
     "usage, no explanation.",
     """
assert roman_to_int("III") == 3
assert roman_to_int("IX") == 9
assert roman_to_int("LVIII") == 58
assert roman_to_int("MCMXCIV") == 1994
assert roman_to_int("MMXXVI") == 2026
print("ALL_OK")
"""),
    ("topwords", 512,
     "Write a Python function named top_words(text, k) that splits text on "
     "whitespace, lowercases the words and strips trailing punctuation characters "
     ".,!?;: from each word, counts them, and returns the k most frequent as a list "
     "of (word, count) tuples sorted by count descending, ties broken "
     "alphabetically. Respond with a single fenced ```python code block containing "
     "only the function definition — no example usage, no explanation.",
     """
assert top_words("The cat and the dog. The dog!", 2) == [("the", 3), ("dog", 2)]
assert top_words("b a b c a", 2) == [("a", 2), ("b", 2)]
assert top_words("one", 3) == [("one", 1)]
assert top_words("Hi, hi! HI? bye", 1) == [("hi", 3)]
print("ALL_OK")
"""),
    ("flatten", 512,
     "Write a Python function named flatten_dict(d) that returns a new dict with "
     "all nested dictionaries flattened into dotted keys (e.g. {\"a\": {\"b\": 1}} "
     "becomes {\"a.b\": 1}); an empty nested dict contributes nothing. Respond with "
     "a single fenced ```python code block containing only the function definition "
     "— no example usage, no explanation.",
     """
assert flatten_dict({"a": {"b": {"c": 1}}, "d": 2}) == {"a.b.c": 1, "d": 2}
assert flatten_dict({"x": {"y": {"z": 5}}, "w": {}}) == {"x.y.z": 5}
assert flatten_dict({}) == {}
assert flatten_dict({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}
print("ALL_OK")
"""),
    ("movingavg", 512,
     "Write a Python class named MovingAvg with __init__(self, window), "
     "push(self, x) and mean(self): mean() returns the average of the most recent "
     "`window` pushed values (or of all pushed values if fewer than `window` so "
     "far). Respond with a single fenced ```python code block containing only the "
     "class definition — no example usage, no explanation.",
     """
m = MovingAvg(3)
m.push(1); m.push(2)
assert m.mean() == 1.5
m.push(3)
assert m.mean() == 2.0
m.push(10)
assert m.mean() == 5.0
print("ALL_OK")
"""),
    ("lowerbound", 512,
     "Write a Python function named lower_bound(arr, x) that returns the first "
     "index i with arr[i] >= x in the sorted list arr (or len(arr) if every element "
     "is smaller). Respond with a single fenced ```python code block containing "
     "only the function definition — no example usage, no explanation.",
     """
assert lower_bound([], 5) == 0
assert lower_bound([1, 3, 3, 7], 3) == 1
assert lower_bound([1, 3, 3, 7], 0) == 0
assert lower_bound([1, 3, 3, 7], 9) == 4
assert lower_bound([1, 3, 3, 7], 4) == 3
print("ALL_OK")
"""),
]

MATH_TASKS = [
    ("printer", 1024,
     "A printer prints 47 pages per minute for the first 12 minutes, then slows to "
     "31 pages per minute for the rest of a 25-minute job. How many pages did it "
     "print in total? Solve step by step, then end your reply with the line "
     "'Answer: <integer>'.",
     967),
    ("profit", 1024,
     "A store buys 80 units at $14 each. It sells 65 of them at $23 each and the "
     "remaining 15 at $9 each. What is the total profit in dollars? Solve step by "
     "step, then end your reply with the line 'Answer: <integer>'.",
     510),
    ("trains", 1024,
     "Train A leaves at 9:00 travelling 90 km/h. Train B leaves the same station at "
     "9:40 on the same track travelling 120 km/h. How many minutes after 9:00 does "
     "Train B catch up to Train A? Solve step by step, then end your reply with the "
     "line 'Answer: <integer>'.",
     160),
    ("tank", 1024,
     "An empty 2,400 litre tank fills at 85 L/min for 20 minutes, then drains at "
     "45 L/min for 12 minutes, then fills at 40 L/min until full. How many minutes "
     "does that final fill take? Solve step by step, then end your reply with the "
     "line 'Answer: <integer>'.",
     31),
    ("sum210", 1024,
     "The sum of all integers from 1 through n equals 210. Find n. Solve step by "
     "step, then end your reply with the line 'Answer: <integer>'.",
     20),
    ("flour", 1024,
     "A recipe for 6 people needs 450 g of flour. You are cooking for 8 people and "
     "have only a 700 g bag. How many grams of flour remain after cooking? Solve "
     "step by step, then end your reply with the line 'Answer: <integer>'.",
     100),
]

FORMAT_TASKS = [
    ("json_cities", 256,
     "Population figures (2024 est.): Brannvik 46,320; Holmstrand 55,123; "
     "Vesterby 63,450; Kaldnes 56,300. Emit ONLY a JSON object of the form "
     "{\"cities\": [{\"name\": \"...\", \"population\": <integer>}, ...]} listing "
     "all four cities sorted by population, largest first. No other text.",
     "json_cities"),
    ("bullets", 256,
     "Explain why prefix caching makes repeated prompts faster. Use exactly 3 "
     "bullet points, each starting with '- ' and each at most 12 words. No other "
     "text before or after.",
     "bullets"),
    ("toolcall", 256,
     "Emit ONLY a JSON object with exactly the keys \"tool\" (string, exactly "
     "\"web_search\") and \"args\" (object with exactly the keys \"query\" "
     "(string, exactly \"ARM NEON dot product instruction\") and \"limit\" "
     "(integer, exactly 5)). No other text.",
     "toolcall"),
    ("hexmix", 256,
     "Average each RGB channel of the colors #3A1F0C and #C24D82 (integer result "
     "per channel), and output ONLY the resulting color as a 7-character hex code "
     "in lowercase, like #a1b2c3. No other text.",
     "hexmix"),
    ("sortlist", 256,
     "Output ONLY the following words sorted alphabetically, lowercase, "
     "comma-separated: Mango, kiwi, APPLE, Pear, Ugli. No other text.",
     "sortlist"),
]

RECALL_TASKS = [
    ("walrus", 200,
     "In one or two sentences: what does Python's walrus operator do?",
     [":=", "express", "assign"]),
    ("heapify", 200,
     "In one or two sentences: what is the time complexity of building a heap from "
     "an unsorted list with Python's heapq.heapify?",
     ["o(n)", "linear"]),
    ("tcpudp", 200,
     "In one or two sentences: what is the key practical difference between TCP and "
     "UDP?",
     ["reliab", "connect"]),
]

# Long-context block: the committed ledger document (c23_eval_doc.txt) + 4
# questions over one appended conversation. Answers computed from the generator.
LONGCTX_QUESTIONS = [
    ("ledger_total",
     "Using only the ledger above: how many servo motors in total were shipped "
     "across all destinations? Sum the quantity of every shipment whose item is "
     "servo motors. End with the line 'Answer: <integer>'.",
     2000, "int"),
    ("ledger_maxcost",
     "Using only the ledger above: which shipment identifier has the highest unit "
     "cost? End with the line 'Answer: <shipment id>'.",
     "SHP-4268", "id"),
    ("ledger_rotapril",
     "Using only the ledger above: how many shipments departed from Rotterdam "
     "during April 2025 (any day in April)? End with the line 'Answer: <integer>'.",
     2, "int"),
    ("ledger_revenue",
     "Using only the ledger above: what is the total revenue in dollars of all "
     "hydraulic pump shipments (each shipment's quantity times its unit cost, "
     "summed)? End with the line 'Answer: <integer>'.",
     2942800, "int"),
]
