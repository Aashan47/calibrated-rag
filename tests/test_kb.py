"""The editable help centre and the human queue — offline.

What must hold: an edit re-indexes the agent immediately; article ids stay stable across
deletions; a run that started before an edit keeps a consistent view; bad input is refused
with a clear message; the shipped help centre can always be restored; write-through never
escapes its directory; and the queue states (escalated → closed) are enforced."""

import os
import tempfile
import unittest

from helpdesk_agent import corpus, kb
from helpdesk_agent.agent import Agent
from helpdesk_agent.inbox import Inbox, TransitionError

ARTS = [
    {"slug": "refund-policy", "title": "Refund policy", "category": "billing",
     "category_name": "Billing & plans",
     "text": "Annual subscriptions can be refunded in full within 14 days of purchase."},
    {"slug": "slack-integration", "title": "Slack integration", "category": "integrations",
     "category_name": "Integrations",
     "text": "Connect Slack from Settings to post channel notifications for new tasks."},
]
NEW_BODY = ("Startup discount: a startup under two years old with fewer than ten employees gets "
            "30% off the Business plan for the first year. Apply from the billing page.")


def factory(contexts):
    return Agent(contexts, k=2, n_samples=1, retriever="tfidf")


def make(**kw):
    return kb.KnowledgeBase([dict(a) for a in ARTS], "test help centre", "Testco",
                            {"billing": "Billing & plans", "integrations": "Integrations"},
                            factory, **kw)


class LiveReindex(unittest.TestCase):
    def test_added_article_is_searchable_on_the_next_run(self):
        k = make()
        before = k.snapshot()
        self.assertEqual(before.agent.search("startup discount"), [])      # nothing matches
        a = k.add("Startup discounts", "billing", NEW_BODY)
        after = k.snapshot()
        hits = after.agent.search("startup discount")
        self.assertEqual(after.ids([h["corpus_id"] for h in hits])[:1], [a["id"]])
        self.assertEqual(after.version, before.version + 1)
        self.assertEqual(after.edits, 1)
        self.assertEqual(a["state"], "added")
        self.assertEqual(before.agent.search("startup discount"), [])      # old snapshot untouched

    def test_update_reindexes_and_marks_edited(self):
        k = make()
        a = k.update(1, text="Annual subscriptions can be refunded within 30 days of purchase.")
        self.assertEqual(a["state"], "edited")
        snap = k.snapshot()
        hit = snap.agent.search("refund 30 days")[0]["corpus_id"]
        self.assertIn("30 days", snap.article(hit)["text"])

    def test_update_without_change_is_a_no_op(self):
        k = make()
        v = k.snapshot().version
        k.update(1, title="Refund policy")
        self.assertEqual(k.snapshot().version, v)
        self.assertEqual(k.edits, 0)

    def test_ids_are_stable_after_delete(self):
        k = make()
        c = k.add("Third", "billing", NEW_BODY)
        k.delete(1)
        snap = k.snapshot()
        self.assertEqual([a["id"] for a in snap.articles], [2, c["id"]])
        hit = snap.agent.search("startup")[0]["corpus_id"]
        self.assertEqual(snap.article(hit)["id"], c["id"])              # position ≠ id
        with self.assertRaises(KeyError):
            k.update(1, title="gone")

    def test_new_category_is_created_and_listed(self):
        k = make()
        a = k.add("Partner programme", "Partnerships", NEW_BODY)
        self.assertEqual(a["category"], "partnerships")
        self.assertEqual(k.categories()["partnerships"], "Partnerships")
        self.assertEqual(a["category_name"], "Partnerships")


class Validation(unittest.TestCase):
    def test_refuses_bad_input_with_a_reason(self):
        k = make()
        bad = [("", "billing", NEW_BODY), ("ok title", "billing", "too short"),
               ("x" * 121, "billing", NEW_BODY), ("ok title", "billing", "y" * 6001),
               ("ok title", "!!!", NEW_BODY), (None, "billing", NEW_BODY),
               ("ok title", "billing", ["not", "text"])]
        for title, cat, text in bad:
            with self.assertRaises(kb.KBError, msg=repr((title, cat, text))[:40]):
                k.add(title, cat, text)
        self.assertEqual(k.edits, 0)
        self.assertEqual(len(k.list()), 2)

    def test_failed_add_does_not_create_a_category(self):
        k = make()
        with self.assertRaises(kb.KBError):
            k.add("Refund policy", "billing", NEW_BODY)          # duplicate title
        with self.assertRaises(kb.KBError):
            k.add("", "brand-new", NEW_BODY)                     # invalid title, new category
        self.assertNotIn("brand-new", k.categories())

    def test_duplicate_title_in_category_refused_but_allowed_elsewhere(self):
        k = make()
        with self.assertRaises(kb.KBError):
            k.add("refund POLICY", "billing", NEW_BODY)
        k.add("Refund policy", "integrations", NEW_BODY)

    def test_control_chars_and_whitespace_normalised(self):
        k = make()
        a = k.add("  Odd \x00 title  ", "billing", "Line one.\r\n\r\n\r\n\r\nLine two with  spaces.")
        self.assertEqual(a["title"], "Odd title")
        self.assertEqual(a["text"], "Line one.\n\nLine two with spaces.")

    def test_cap_and_non_empty(self):
        k = make(max_articles=3)
        k.add("Third", "billing", NEW_BODY)
        with self.assertRaises(kb.KBError):
            k.add("Fourth", "billing", NEW_BODY)
        k.delete(1); k.delete(2)
        with self.assertRaises(kb.KBError):
            k.delete(3)                                          # never empty


class ResetAndExport(unittest.TestCase):
    def test_reset_restores_shipped_and_counts_discards(self):
        k = make()
        k.add("Added article", "billing", NEW_BODY); k.update(2, text=NEW_BODY); k.delete(1)
        self.assertEqual(k.reset(), 3)
        snap = k.snapshot()
        self.assertEqual([(a["id"], a["title"], a["state"]) for a in snap.articles],
                         [(1, "Refund policy", None), (2, "Slack integration", None)])
        self.assertEqual(snap.edits, 0)
        self.assertEqual(k.reset(), 0)
        hit = snap.agent.search("annual subscriptions refunded")[0]["corpus_id"]
        self.assertIn("14 days", snap.article(hit)["text"])

    def test_export_round_trips_through_load_knowledge_base(self):
        k = make()
        k.add("Startup discounts", "billing", NEW_BODY)
        e = k.export()
        self.assertEqual({a["title"] for a in e["articles"]},
                         {"Refund policy", "Slack integration", "Startup discounts"})
        self.assertEqual(set(e["categories"]), {"billing", "integrations"})


class WriteThrough(unittest.TestCase):
    def test_edits_are_mirrored_as_markdown_and_never_escape_the_directory(self):
        with tempfile.TemporaryDirectory() as d:
            k = make(write_dir=d)
            a = k.add("../../etc/passwd", "../billing", NEW_BODY)
            path = os.path.join(d, "billing", "etc-passwd.md")
            self.assertTrue(os.path.exists(path), os.listdir(d))
            self.assertFalse(os.path.exists(os.path.join(os.path.dirname(d), "etc")))
            loaded = corpus.load_knowledge_base(d)
            self.assertEqual(loaded["articles"][0]["title"], "../../etc/passwd")
            self.assertEqual(loaded["articles"][0]["text"], NEW_BODY)
            k.delete(a["id"])
            self.assertFalse(os.path.exists(path))

    def test_write_failure_does_not_break_the_edit(self):
        k = make(write_dir="/nonexistent/dir/that/cannot/be/created\x00")
        a = k.add("Still works", "billing", NEW_BODY)
        self.assertEqual(len(k.list()), 3)
        self.assertEqual(a["title"], "Still works")


class HumanQueue(unittest.TestCase):
    def test_escalated_can_be_closed_or_rerun_and_closed_is_final(self):
        ib = Inbox(seed_path=None)
        t = ib.add("Do you offer startup discounts?", subject="Discount")
        ib.set_status(t["id"], "handling")
        ib.set_status(t["id"], "escalated", result={"status": "escalated",
                                                   "reason_code": "not_in_documents",
                                                   "reason_title": "Not covered",
                                                   "handoff": {"closest": []}})
        q = ib.list("escalated")
        self.assertEqual([x["id"] for x in q], [t["id"]])
        self.assertEqual(q[0]["outcome"]["handoff"], {"closest": []})
        self.assertEqual(q[0]["outcome"]["reason_title"], "Not covered")
        ib.set_status(t["id"], "new")                            # re-run after a KB fix
        self.assertEqual(ib.list("escalated"), [])
        ib.set_status(t["id"], "escalated")
        pub = ib.set_status(t["id"], "closed")
        self.assertIsNotNone(pub["closed_at"])
        self.assertEqual(ib.counts()["closed"], 1)
        for nxt in ("new", "handling", "escalated", "sent"):
            with self.assertRaises(TransitionError):
                ib.set_status(t["id"], nxt)

    def test_only_escalated_tickets_can_be_closed(self):
        ib = Inbox(seed_path=None)
        t = ib.add("hello")
        with self.assertRaises(TransitionError):
            ib.set_status(t["id"], "closed")

    def test_closed_counts_as_finished_for_the_cap(self):
        ib = Inbox(seed_path=None, max_tickets=1)
        t = ib.add("a")
        ib.set_status(t["id"], "escalated"); ib.set_status(t["id"], "closed")
        ib.add("b")
        self.assertEqual(ib.counts()["total"], 1)


if __name__ == "__main__":
    unittest.main()
