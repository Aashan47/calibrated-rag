"""Guardrails: input, load, output grounding, inbox transitions, retrieval on no-match."""

import time
import unittest

from helpdesk_agent import guards
from helpdesk_agent.inbox import Inbox, TransitionError
from helpdesk_agent.retriever import TfidfRetriever


class InputGuard(unittest.TestCase):
    def test_normalises_whitespace_and_strips_control_chars(self):
        self.assertEqual(guards.clean_message("  hi\x00 there \r\n\r\n\r\n ok  "), "hi there\n\nok")

    def test_rejects_empty_oversized_and_non_text(self):
        for bad in ["", "   ", "\x00", "x" * (guards.MAX_MESSAGE_CHARS + 1), None, 42, ["a"]]:
            with self.assertRaises(guards.MessageError, msg=repr(bad)[:30]):
                guards.clean_message(bad)

    def test_keeps_unicode(self):
        self.assertEqual(guards.clean_message("Können wir in € zahlen? 🙂"), "Können wir in € zahlen? 🙂")


class OutputGrounding(unittest.TestCase):
    ART = ("Annual subscriptions can be refunded in full if you request the refund within 14 "
           "days of the purchase or renewal date.")

    def test_fact_in_article_is_grounded(self):
        self.assertTrue(guards.grounded("Yes, within 14 days of purchase", self.ART))
        self.assertTrue(guards.grounded("14 days", self.ART))

    def test_fact_from_nowhere_is_not(self):
        self.assertFalse(guards.grounded("30 days", self.ART))
        self.assertFalse(guards.grounded("Yes, 14 days or $50", self.ART))   # one number invented
        self.assertFalse(guards.grounded("Yes, we offer a startup programme", self.ART))

    def test_bare_yes_no_passes(self):
        self.assertTrue(guards.grounded("Yes.", self.ART))
        self.assertFalse(guards.grounded("", self.ART))


class LoadGuards(unittest.TestCase):
    def test_rate_limiter_bucket(self):
        rl = guards.RateLimiter(rate_per_min=60, burst=3)
        t0 = 1000.0
        self.assertEqual([rl.allow("a", t0) for _ in range(4)], [True, True, True, False])
        self.assertTrue(rl.allow("b", t0))                 # other client unaffected
        self.assertTrue(rl.allow("a", t0 + 1.1))           # one token refilled after ~1s

    def test_gate_does_not_queue(self):
        g = guards.Gate(1)
        self.assertTrue(g.acquire())
        self.assertFalse(g.acquire())
        g.release()
        self.assertTrue(g.acquire())


class InboxTransitions(unittest.TestCase):
    def test_cannot_send_an_unresolved_ticket(self):
        ib = Inbox(seed_path=None)
        t = ib.add("hello")
        with self.assertRaises(TransitionError):
            ib.set_status(t["id"], "sent")

    def test_cannot_handle_a_ticket_twice_concurrently(self):
        ib = Inbox(seed_path=None)
        t = ib.add("hello")
        ib.set_status(t["id"], "handling")
        with self.assertRaises(TransitionError):
            ib.set_status(t["id"], "handling")

    def test_stale_handling_is_reclaimable(self):
        ib = Inbox(seed_path=None)
        t = ib.add("hello")
        ib.set_status(t["id"], "handling")
        ib._tickets[t["id"]]["started_at"] = time.time() - 10_000
        ib.set_status(t["id"], "handling")                 # no exception: reclaimed

    def test_sent_is_final_and_reopen_clears_result(self):
        ib = Inbox(seed_path=None)
        t = ib.add("hello")
        ib.set_status(t["id"], "handling")
        ib.set_status(t["id"], "auto-resolved", result={"status": "resolved", "reply": "ok"})
        ib.set_status(t["id"], "new")
        self.assertIsNone(ib.get(t["id"])["result"])
        ib.set_status(t["id"], "handling")
        ib.set_status(t["id"], "auto-resolved", result={"status": "resolved", "reply": "ok"})
        ib.set_status(t["id"], "sent")
        with self.assertRaises(TransitionError):
            ib.set_status(t["id"], "new")

    def test_cap_drops_finished_not_live(self):
        ib = Inbox(seed_path=None, max_tickets=2)
        a = ib.add("a"); b = ib.add("b")
        with self.assertRaises(TransitionError):
            ib.add("c")                                   # both live → refuse
        ib.set_status(a["id"], "escalated")
        ib.add("c")                                       # drops the finished one
        self.assertEqual(ib.counts()["total"], 2)


class RetrievalNoMatch(unittest.TestCase):
    def test_no_overlap_returns_nothing(self):
        r = TfidfRetriever(["refund policy annual plan", "slack integration channels"])
        self.assertEqual(r.search("zzzz qqqq", k=3), [])
        self.assertEqual(r.search("", k=3), [])
        self.assertEqual(r.search("refund", k=3), [0])


if __name__ == "__main__":
    unittest.main()
