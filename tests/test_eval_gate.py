"""End-to-end regression ('eval gate') tests — offline, with a deterministic mock LLM.

Unlike test_core.py (which unit-tests the loop's control flow with scripted decisions), this
runs the *whole agent* — real retriever, real loop, real aggregation — against a small golden
corpus with a content-aware mock model, and asserts the behaviours a deployment must not
regress: it answers known questions with a citation to the *correct* passage, abstains on
out-of-corpus questions, the loop always terminates, and single-shot mode does exactly one
retrieval. Safe for CI (no API key needed).

    python -m unittest tests.test_eval_gate -v
"""

import re
import unittest

from helpdesk_agent import agent as agent_mod
from helpdesk_agent import llm
from helpdesk_agent.agent import Agent

CORPUS = [
    "Paris is the capital of France and its largest city.",   # 0
    "The Eiffel Tower, a landmark, is located in Paris.",     # 1
    "Mount Everest is the tallest mountain above sea level.", # 2
    "Photosynthesis converts light into chemical energy.",    # 3
]


def _gold(question: str):
    q = question.lower()
    if "france" in q:
        return "Paris"
    if "tallest" in q or "everest" in q:
        return "Everest"
    if "photosynthesis" in q:
        return "light"
    return None   # out of corpus -> should abstain


class MockLLM:
    """Content-aware stand-in: 'answers' only when the gold token is actually in the passages."""

    @staticmethod
    def complete(prompt, **kw):
        q = re.search(r"QUESTION:\s*(.+)", prompt)
        gold = _gold(q.group(1)) if q else None
        passages = prompt.split("PASSAGES GATHERED SO FAR:")[-1]
        if gold and gold.lower() in passages.lower():
            return '{"action":"answer","reason":"found"}'
        return '{"action":"abstain","reason":"not in corpus"}'

    @staticmethod
    def answer_or_abstain(question, context, temperature=0.0):
        gold = _gold(question)
        if not gold:
            return {"answerable": False, "answer": "", "cite": 0, "confidence": 0.0}
        for block in context.split("\n\n"):           # blocks look like "[3] text..."
            m = re.match(r"\[(\d+)\]\s*(.*)", block, re.S)
            if m and gold.lower() in m.group(2).lower():
                return {"answerable": True, "answer": gold,
                        "cite": int(m.group(1)), "confidence": 1.0}
        return {"answerable": False, "answer": "", "cite": 0, "confidence": 0.0}


class EvalGate(unittest.TestCase):
    def setUp(self):
        self._orig = (llm.complete, llm.answer_or_abstain)
        llm.complete = MockLLM.complete
        llm.answer_or_abstain = MockLLM.answer_or_abstain
        agent_mod.llm.answer_or_abstain = MockLLM.answer_or_abstain

    def tearDown(self):
        llm.complete, llm.answer_or_abstain = self._orig
        agent_mod.llm.answer_or_abstain = self._orig[1]

    def agent(self, max_steps=3):
        return Agent(CORPUS, k=2, n_samples=3, max_steps=max_steps, retriever="tfidf")

    def test_answers_with_correct_citation(self):
        p = self.agent().predict("What is the capital of France?")
        self.assertTrue(p["answerable"])
        self.assertEqual(p["answer"], "Paris")
        self.assertEqual(p["confidence"], 1.0)
        self.assertIsNotNone(p["citation"])
        self.assertEqual(p["citation"]["corpus_id"], 0)        # grounded in the France passage

    def test_abstains_out_of_corpus(self):
        p = self.agent().predict("What is the capital of Mars?")
        self.assertFalse(p["answerable"])
        self.assertEqual(p["confidence"], 0.0)
        self.assertTrue(any(s["action"] == "abstain" for s in p["steps"]))

    def test_second_fact_also_grounded(self):
        p = self.agent().predict("What is the tallest mountain?")
        self.assertTrue(p["answerable"])
        self.assertEqual(p["citation"]["corpus_id"], 2)

    def test_loop_is_bounded(self):
        # even if the model never says "answer", the loop must terminate
        llm.complete = lambda prompt, **kw: '{"action":"search","query":"x"+str(id(prompt))}'
        p = self.agent(max_steps=3).predict("What is the capital of France?")
        searches = [s for s in p["steps"] if s["action"] == "search"]
        self.assertLessEqual(len(searches), 1 + 3)             # seed + at most max_steps

    def test_single_shot_does_one_retrieval(self):
        p = self.agent(max_steps=0).predict("What is the capital of France?")
        searches = [s for s in p["steps"] if s["action"] == "search"]
        self.assertEqual(len(searches), 1)                     # no decision loop
        self.assertTrue(p["answerable"])                       # still answers


if __name__ == "__main__":
    unittest.main()
