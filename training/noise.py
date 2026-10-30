"""Synthetic grammar errors: turn a correct sentence into the kind of text people actually type.

Pure Python, no dependencies. `python noise.py` prints examples and runs a self-check.
Each error function takes (sentence, rng) and returns a corrupted sentence or None if it does not apply.
"""

import random
import re


def _keep_case(src, word):
    return word[:1].upper() + word[1:] if src[:1].isupper() else word


def _swap(s, rng, table):
    """Replace one word found in `table` (right -> [wrong, ...]) with a wrong variant, preserving capitalization."""
    hits = [(m, right) for right in table for m in re.finditer(rf"\b{re.escape(right)}\b", s, re.I)]
    if not hits:
        return None
    m, right = rng.choice(hits)
    wrong = rng.choice(table[right])
    return s[: m.start()] + _keep_case(m.group(0), wrong) + s[m.end():]


# Only swaps that are always wrong: a<->an, or a dropped article. "a" -> "the" can still be correct English.
ARTICLES = {"a": ["an", ""], "an": ["a", ""], "the": [""]}
PREPOSITIONS = {p: [q for q in ["in", "on", "at", "to", "for", "of", "with", "from", "by", "about"] if q != p]
                for p in ["in", "on", "at", "to", "for", "of", "with", "from", "by", "about"]}
AGREEMENT = {"is": ["are"], "are": ["is"], "was": ["were"], "were": ["was"], "has": ["have"], "have": ["has"],
             "does": ["do"], "doesn't": ["don't"], "don't": ["doesn't"], "am": ["is", "are"]}
# base: (3rd person, past, participle)
VERBS = {"go": ("goes", "went", "gone"), "come": ("comes", "came", "come"), "see": ("sees", "saw", "seen"),
         "do": ("does", "did", "done"), "eat": ("eats", "ate", "eaten"), "write": ("writes", "wrote", "written"),
         "take": ("takes", "took", "taken"), "give": ("gives", "gave", "given"), "know": ("knows", "knew", "known"),
         "begin": ("begins", "began", "begun"), "break": ("breaks", "broke", "broken"), "speak": ("speaks", "spoke", "spoken"),
         "run": ("runs", "ran", "run"), "drink": ("drinks", "drank", "drunk"), "choose": ("chooses", "chose", "chosen"),
         "make": ("makes", "made", "made"), "get": ("gets", "got", "gotten"), "think": ("thinks", "thought", "thought"),
         "buy": ("buys", "bought", "bought"), "send": ("sends", "sent", "sent"), "need": ("needs", "needed", "needed"),
         "want": ("wants", "wanted", "wanted"), "like": ("likes", "liked", "liked"), "work": ("works", "worked", "worked"),
         "use": ("uses", "used", "used"), "help": ("helps", "helped", "helped"), "look": ("looks", "looked", "looked"),
         "say": ("says", "said", "said"), "tell": ("tells", "told", "told"), "feel": ("feels", "felt", "felt")}
TENSE = {}
for base, (third, past, part) in VERBS.items():
    for form in {base, third, past, part}:
        TENSE[form] = sorted({base, third, past, part} - {form})
HOMOPHONES = {"your": ["you're"], "you're": ["your"], "its": ["it's"], "it's": ["its"], "their": ["there", "they're"],
              "there": ["their"], "they're": ["their", "there"], "to": ["too"], "too": ["to"], "than": ["then"],
              "then": ["than"], "lose": ["loose"], "affect": ["effect"], "effect": ["affect"], "whose": ["who's"],
              "who's": ["whose"], "where": ["were"], "were": ["where"], "accept": ["except"], "quite": ["quiet"]}
PLURALS = {"people": ["peoples", "person"], "children": ["childrens", "child"], "information": ["informations"],
           "advice": ["advices"], "furniture": ["furnitures"], "equipment": ["equipments"], "news": ["new"],
           "feedback": ["feedbacks"], "homework": ["homeworks"], "research": ["researches"]}
MODAL_OF = {"could have": ["could of"], "would have": ["would of"], "should have": ["should of"], "must have": ["must of"]}
FUNCTION_WORDS = ["the", "a", "an", "to", "is", "are", "of", "that", "and", "be", "was", "it", "for"]
CONTRACTIONS = ["don't", "doesn't", "didn't", "can't", "won't", "isn't", "aren't", "wasn't", "weren't", "it's",
                "that's", "what's", "there's", "I'm", "you're", "they're", "we're", "I've", "I'll", "let's", "he's",
                "she's", "couldn't", "wouldn't", "shouldn't", "haven't", "hasn't"]


def article(s, rng): return _swap(s, rng, ARTICLES)
def preposition(s, rng): return _swap(s, rng, PREPOSITIONS)
def agreement(s, rng): return _swap(s, rng, AGREEMENT)
def tense(s, rng): return _swap(s, rng, TENSE)
def homophone(s, rng): return _swap(s, rng, HOMOPHONES)
def plural(s, rng): return _swap(s, rng, PLURALS)
def modal_of(s, rng): return _swap(s, rng, MODAL_OF)


def apostrophes(s, rng):
    out = s
    for c in CONTRACTIONS:
        out = re.sub(rf"\b{re.escape(c)}\b", lambda m: m.group(0).replace("'", ""), out, flags=re.I)
    return out if out != s else None


def drop_word(s, rng):
    return _swap(s, rng, {w: [""] for w in FUNCTION_WORDS})


def duplicate_word(s, rng):
    words = s.split(" ")
    if len(words) < 4:
        return None
    i = rng.randrange(1, len(words) - 1)
    return " ".join(words[: i + 1] + [words[i]] + words[i + 1:])


def word_order(s, rng):
    words = s.split(" ")
    if len(words) < 5:
        return None
    i = rng.randrange(1, len(words) - 2)
    if not (words[i].isalpha() and words[i + 1].isalpha()):
        return None
    words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def typo(s, rng):
    words = [(m.start(), m.group(0)) for m in re.finditer(r"[a-z]{5,}", s)]
    if not words:
        return None
    at, w = rng.choice(words)
    i = rng.randrange(1, len(w) - 1)
    bad = rng.choice([w[:i] + w[i + 1:], w[:i] + w[i + 1] + w[i] + w[i + 2:], w[:i] + w[i] + w[i:]])
    return s[:at] + bad + s[at + len(w):]


def lowercase(s, rng):
    out = re.sub(r"\bI\b", "i", s) if rng.random() < 0.5 else s
    out = out[:1].lower() + out[1:]
    return out if out != s else None


def punctuation(s, rng):
    out = s
    if "," in out and rng.random() < 0.6:
        out = out.replace(", ", " ")
    if out[-1:] in ".?!" and rng.random() < 0.7:
        out = out[:-1]
    return out if out != s else None


def comparative(s, rng):
    """taller than -> more taller than, the tallest -> the most tallest"""
    m = re.search(r"\b([a-z]{3,}er) than\b", s) or re.search(r"\bthe ([a-z]{3,}est)\b", s)
    if not m or m.group(1) in {"other", "rather", "never", "over", "under", "after", "interest", "forest", "request",
                                "honest", "latest", "rest", "test", "best"}:
        return None
    word = "more " if m.group(0).endswith("than") else "most "
    at = m.start(1)
    return s[:at] + word + s[at:]


ERRORS = [article, preposition, agreement, tense, homophone, plural, modal_of, apostrophes, drop_word,
          duplicate_word, word_order, typo, lowercase, punctuation, comparative]
WEIGHTS = [8, 3, 9, 9, 10, 2, 2, 8, 5, 2, 2, 3, 6, 9, 3]  # prepositions low: a swap is sometimes still correct


def corrupt(s, rng, max_errors=3):
    """Apply 1..max_errors different errors. Returns None if none applied."""
    out = s
    n = rng.choice([1, 1, 2, 2, 3][: max_errors + 2])
    tried = set()
    while n and len(tried) < len(ERRORS):
        fn = rng.choices(ERRORS, WEIGHTS)[0]
        if fn in tried:
            continue
        tried.add(fn)
        new = fn(out, rng)
        if new and new != out:
            out = re.sub(r" {2,}", " ", new).strip()
            n -= 1
    return out if out != s and out else None


def run_on(sentences, rng):
    """Glue 2-3 correct sentences together the way people type in chat."""
    parts = []
    for i, s in enumerate(sentences):
        p = s[:-1] if rng.random() < 0.9 else s
        if i > 0 and not re.match(r"I\b", p):
            p = p[:1].lower() + p[1:]
        parts.append(p)
    joiner = rng.choice([" ", " ", " and ", ", "])
    src = joiner.join(parts)
    return src, " ".join(sentences)


SUBJECTS = [("she", "She"), ("he", "He"), ("they", "They"), ("my mom", "My mom"), ("the teacher", "The teacher"),
            ("my friend", "My friend"), ("I", "I"), ("he just", "He just"), ("my boss", "My boss"), ("someone", "Someone"),
            ("the kid", "The kid"), ("she literally", "She literally"), ("my dad", "My dad"), ("he then", "He then")]
SPEECH_VERBS = ["said", "asked", "told me", "kept saying", "yelled", "replied", "whispered", "texted me"]
QUESTIONS = ["what's all the fuss about", "why are you so late", "where did you put my keys", "what is going on here",
             "how did you do that", "who told you that", "when are we leaving", "can you help me with this",
             "are you coming tonight", "is this your bag", "do you even care", "what time is it",
             "why is everyone so upset", "have you finished your homework", "what's wrong with you", "did you eat",
             "where are you going", "how much does it cost", "can I come too", "what's for dinner",
             "is it raining outside", "why didn't you call me", "who's going to pay for this", "are we there yet"]
STATEMENTS = ["I'm not coming tonight", "this is the best day ever", "we need to talk", "the food was amazing",
              "don't touch my phone", "you're going to love it", "I can't believe it", "it's not my fault",
              "I'll be there in five minutes", "that's not fair", "we're out of milk", "I already told you",
              "you have to try this", "it doesn't matter anymore", "the meeting is cancelled", "I love this song"]


def speech(rng):
    """She said what's all the fuss about  ->  She said, "What's all the fuss about?\""""
    low, cap = rng.choice(SUBJECTS)
    verb = rng.choice(SPEECH_VERBS)
    if rng.random() < 0.6:
        q, end = rng.choice(QUESTIONS), "?"
    else:
        q, end = rng.choice(STATEMENTS), rng.choice([".", ".", "!"])
    quoted = q[:1].upper() + q[1:] + end
    typed = apostrophes(q, rng) or q if rng.random() < 0.4 else q
    src = f"{low if rng.random() < 0.6 else cap} {verb} {typed}"
    if rng.random() < 0.2:
        src += end
    return src, f'{cap} {verb}, "{quoted}"'


REPORTED = ["what the fuss was about", "where I was going", "why I was so late", "what time it was", "who had told me that",
            "if I was coming home tonight", "whether we had finished", "how much it cost", "what was wrong with me",
            "where she had put the keys", "if I could help him", "what we were having for dinner", "why nobody called",
            "how I did that", "when we were leaving", "if it was raining outside", "what the plan was", "who was paying"]


def reported(rng):
    """Indirect speech stays unquoted: "She asked what the fuss was about." is already correct.
    Half are returned untouched, half typed chat-style and fixed only in case and full stop."""
    low, cap = rng.choice(SUBJECTS)
    verb = rng.choice(["asked", "asked me", "wanted to know", "wondered", "asked us"])
    clause = rng.choice(REPORTED)
    if low == "I" and "I " in clause:
        clause = clause.replace("I ", "she ", 1)
    good = f"{cap} {verb} {clause}."
    if rng.random() < 0.5:
        return good, good
    return f"{low} {verb} {clause}", good


if __name__ == "__main__":
    rng = random.Random(1)
    samples = ["She goes to school every day.", "I have seen that movie before.", "It's not your fault, and you know it.",
               "They were happy with the information we sent.", "He could have told me about the meeting.",
               "My brother is taller than me."]
    for fn in ERRORS:
        outs = [o for o in (fn(s, rng) for s in samples) if o]
        assert outs, f"{fn.__name__} never applied"
        print(f"{fn.__name__:15} {outs[0]}")
    for _ in range(3):
        print("corrupt        ", corrupt(rng.choice(samples), rng))
    print("run_on         ", run_on(samples[:2], rng))
    print("speech         ", speech(rng))
    print("reported       ", reported(rng))
    assert comparative("She is the tallest girl.", rng) == "She is the most tallest girl."
    assert '"' not in "".join(reported(rng)[1] for _ in range(50))
    assert all(corrupt(s, rng) != s for s in samples)
    print("self-check OK")
