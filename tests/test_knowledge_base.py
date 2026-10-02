"""The committed help centre and its labelled tickets must stay consistent — the shipped
threshold is calibrated on them, so a broken article/ticket link silently breaks the guarantee."""

import unittest

from helpdesk_agent import corpus, datasets, metrics
from helpdesk_agent.retriever import TfidfRetriever


class KnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.kb = corpus.load_knowledge_base()
        self.contexts, self.items = datasets.load("helpdesk")

    def test_articles_have_titles_categories_and_sane_length(self):
        self.assertGreaterEqual(len(self.kb["articles"]), 25)
        for a in self.kb["articles"]:
            self.assertTrue(a["title"] and a["category_name"], a["slug"])
            n = len(a["text"].split())
            self.assertTrue(40 <= n <= 220, f"{a['slug']}: {n} words")

    def test_every_answerable_ticket_points_at_an_article_containing_an_answer(self):
        for it in self.items:
            if it["is_impossible"]:
                self.assertIsNone(it["gold_ctx"])
                continue
            self.assertIsNotNone(it["gold_ctx"], it["question"])
            text = metrics._normalize(self.contexts[it["gold_ctx"]])
            # at least one accepted answer should be textually grounded in the gold article
            # (yes/no answers are grounded by the article's statement, so skip those)
            ans = [metrics._normalize(a) for a in it["answers"]]
            if all(a.split()[:1] in (["yes"], ["no"]) for a in ans):
                continue
            self.assertTrue(any(a and a in text for a in ans),
                            f"{it['question']} -> none of {it['answers']} in gold article")

    def test_lexical_retrieval_finds_the_gold_article_for_most_tickets(self):
        r = TfidfRetriever(self.contexts)
        ans = [it for it in self.items if not it["is_impossible"]]
        hits = sum(1 for it in ans if it["gold_ctx"] in r.search(it["question"], k=3))
        self.assertGreaterEqual(hits / len(ans), 0.85, f"recall@3 = {hits}/{len(ans)}")

    def test_default_corpus_is_the_help_centre(self):
        c = corpus.load_corpus()
        if not c["is_custom"]:
            self.assertEqual(len(c["contexts"]), len(self.kb["articles"]))
            self.assertEqual(c["articles"][0]["title"], self.kb["articles"][0]["title"])


if __name__ == "__main__":
    unittest.main()
