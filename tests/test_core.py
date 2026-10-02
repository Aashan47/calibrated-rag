"""Offline unit tests for the core logic (no API calls) — safe to run in CI.

    python -m unittest discover -s tests -v
"""

import unittest

from helpdesk_agent import conformal, embeddings, llm, metrics
from helpdesk_agent import agent as agent_mod
from helpdesk_agent.agent import Agent, _aggregate
from helpdesk_agent.retriever import HybridRetriever, TfidfRetriever


class TestRetriever(unittest.TestCase):
    def test_ranks_relevant_doc_first(self):
        docs = ["the cat sat on the mat",
                "quarterly revenue grew twenty percent",
                "photosynthesis converts light into chemical energy"]
        r = TfidfRetriever(docs)
        self.assertEqual(r.search("how much did revenue grow", k=1)[0], 1)
        self.assertEqual(r.search("what does photosynthesis do", k=1)[0], 2)


class TestHybridRetriever(unittest.TestCase):
    def test_falls_back_to_lexical_without_embeddings(self):
        orig = embeddings.available
        embeddings.available = lambda: False
        try:
            hr = HybridRetriever(["alpha alpha", "beta beta"])
            self.assertEqual(hr.mode, "lexical")
            self.assertEqual(hr.search("alpha", 1)[0], 0)
        finally:
            embeddings.available = orig

    def test_rrf_surfaces_both_lexical_and_dense_winners(self):
        docs = ["alpha alpha", "beta beta", "gamma gamma"]
        vecs = {"alpha alpha": [1., 0, 0], "beta beta": [0, 1., 0], "gamma gamma": [0, 0, 1.],
                "alpha": [0, 0, 1.]}   # the query embeds closest to doc 2 (gamma)
        o = (embeddings.available, embeddings.embed, embeddings.embed_many)
        embeddings.available = lambda: True
        embeddings.embed = lambda t: vecs.get(t, [0, 0, 1.])
        embeddings.embed_many = lambda ts, persist=True: [vecs[t] for t in ts]
        try:
            hr = HybridRetriever(docs)
            self.assertEqual(hr.mode, "hybrid")
            top2 = set(hr.search("alpha", 2))      # lexical picks doc0, dense picks doc2
            self.assertEqual(top2, {0, 2})         # RRF surfaces both
        finally:
            embeddings.available, embeddings.embed, embeddings.embed_many = o


class TestMetrics(unittest.TestCase):
    def test_exact_match_and_f1(self):
        self.assertEqual(metrics.exact_match("the Paris", ["Paris"]), 1)   # articles stripped
        self.assertEqual(metrics.exact_match("London", ["Paris"]), 0)
        self.assertAlmostEqual(metrics.f1("Alexandre Yersin", ["Yersin"]), 2 / 3, places=3)
        self.assertTrue(metrics.is_correct("Alexandre Yersin", ["Alexandre Yersin"]))

    def test_ece(self):
        # two confident predictions, one right one wrong -> |0.5 - 0.9| = 0.4
        self.assertAlmostEqual(metrics.ece([0.9, 0.9], [True, False]), 0.4, places=3)
        # perfectly calibrated
        self.assertAlmostEqual(metrics.ece([1.0, 1.0], [True, True]), 0.0, places=3)


class TestConformal(unittest.TestCase):
    def test_calibrate_finds_threshold(self):
        records = [(0.9, True, True), (0.9, True, True), (0.5, True, False), (0.5, True, True)]
        self.assertEqual(conformal.calibrate(records, alpha=0.1), 0.9)

    def test_calibrate_abstains_when_unachievable(self):
        records = [(0.9, True, False), (0.5, True, False)]
        self.assertEqual(conformal.calibrate(records, alpha=0.1), 1.01)

    def test_selective_report(self):
        records = [(0.9, True, True), (0.4, True, False)]
        rep = conformal.selective_report(records, 0.5)
        self.assertEqual(rep["answered"], 1)
        self.assertAlmostEqual(rep["selective_accuracy"], 1.0)


class TestAggregate(unittest.TestCase):
    def test_self_consistency_confidence_and_citation(self):
        samples = [{"answerable": True, "answer": "Paris", "cite": 1, "confidence": 1.0}] * 3 + \
                  [{"answerable": False, "answer": "", "cite": 0, "confidence": 0.0}] * 2
        agg = _aggregate(samples, 5)
        self.assertTrue(agg["answerable"])
        self.assertEqual(agg["answer"], "Paris")
        self.assertAlmostEqual(agg["confidence"], 0.6)   # 3 of 5 agree
        self.assertEqual(agg["cite_local"], 1)

    def test_all_abstain(self):
        samples = [{"answerable": False, "answer": "", "cite": 0, "confidence": 0.0}] * 5
        agg = _aggregate(samples, 5)
        self.assertFalse(agg["answerable"])
        self.assertEqual(agg["confidence"], 0.0)


class TestAgentLoop(unittest.TestCase):
    """The agent's decision loop (offline: stub the two LLM calls)."""

    DOCS = ["the cat sat on the mat",
            "quarterly revenue grew twenty percent last year",
            "photosynthesis converts light into chemical energy"]

    def _agent(self):
        return Agent(self.DOCS, k=1, n_samples=2, retriever="tfidf")

    def _patch(self, decisions, answerable=True, answer="twenty percent"):
        """Stub llm.decide-sequence (via complete) and the final answer sampling."""
        self._decs = list(decisions)
        def fake_complete(prompt, **kw):
            return self._decs.pop(0) if self._decs else '{"action":"answer"}'
        def fake_answer(q, ctx, temperature=0.0):
            return {"answerable": answerable, "answer": answer if answerable else "",
                    "cite": 1, "confidence": 1.0 if answerable else 0.0}
        self._orig = (llm.complete, llm.answer_or_abstain)
        llm.complete = fake_complete
        agent_mod.llm.answer_or_abstain = fake_answer
        llm.answer_or_abstain = fake_answer

    def tearDown(self):
        if hasattr(self, "_orig"):
            llm.complete, llm.answer_or_abstain = self._orig
            agent_mod.llm.answer_or_abstain = self._orig[1]

    def test_answers_immediately_when_passages_sufficient(self):
        self._patch(['{"action":"answer","reason":"found it"}'])
        p = self._agent().predict("how much did revenue grow")
        self.assertTrue(p["answerable"])
        self.assertEqual(p["answer"], "twenty percent")
        searches = [s for s in p["steps"] if s["action"] == "search"]
        self.assertEqual(len(searches), 1)        # only the seed search

    def test_reformulates_and_searches_again(self):
        # first decision: search with a new query; second: answer
        self._patch(['{"action":"search","query":"company earnings growth"}',
                     '{"action":"answer"}'])
        p = self._agent().predict("how did the business do")
        searches = [s for s in p["steps"] if s["action"] == "search"]
        self.assertEqual(len(searches), 2)        # seed + one reformulation
        self.assertTrue(p["answerable"])

    def test_abstains_when_agent_gives_up(self):
        self._patch(['{"action":"abstain","reason":"not in corpus"}'])
        p = self._agent().predict("who is the president of mars")
        self.assertFalse(p["answerable"])
        self.assertEqual(p["confidence"], 0.0)
        self.assertTrue(any(s["action"] == "abstain" for s in p["steps"]))

    def test_model_outage_is_reported_as_error_not_abstention(self):
        # decision call AND every answer sample fail -> the prediction must say "error",
        # the trace must not claim a judgement was made, and decide() must not answer.
        def failing_complete(prompt, **kw):
            raise llm.LLMError("rate limited by the model API (HTTP 429)")
        def failing_answer(q, ctx, temperature=0.0):
            return {"answerable": False, "answer": "", "cite": 0, "confidence": 0.0,
                    "error": "rate limited by the model API (HTTP 429)"}
        self._orig = (llm.complete, llm.answer_or_abstain)
        llm.complete = failing_complete
        llm.answer_or_abstain = failing_answer
        p = self._agent().predict("how much did revenue grow")
        self.assertIn("429", p["error"])
        self.assertEqual(p["errors"], 2)                    # both samples failed
        dec = [s for s in p["steps"] if s["action"] == "decide"][0]
        self.assertIn("error", dec)                          # trace is honest about it
        self.assertFalse(agent_mod.decide(p, 0.0)["answered"])
        self.assertEqual(p["llm_calls"], 3)                  # 1 decision + 2 samples

    def test_does_not_repeat_a_tried_query(self):
        # agent keeps proposing the SAME query; loop must stop and answer, not spin
        self._patch(['{"action":"search","query":"how much did revenue grow"}'] * 5)
        p = self._agent().predict("how much did revenue grow")
        searches = [s for s in p["steps"] if s["action"] == "search"]
        self.assertEqual(len(searches), 1)        # duplicate query rejected → no re-search


if __name__ == "__main__":
    unittest.main()
