"""The inbox workflow and the agent's live event stream — offline."""

import unittest

from helpdesk_agent import llm
from helpdesk_agent import agent as agent_mod
from helpdesk_agent.agent import Agent
from helpdesk_agent.inbox import Inbox


class InboxWorkflow(unittest.TestCase):
    def test_seed_loads_and_orders_newest_first(self):
        ib = Inbox()
        ts = ib.list()
        self.assertGreaterEqual(len(ts), 10)
        self.assertTrue(all(t["status"] == "new" for t in ts))
        self.assertTrue(all(ts[i]["received"] >= ts[i + 1]["received"] for i in range(len(ts) - 1)))

    def test_lifecycle_new_handling_resolved_sent(self):
        ib = Inbox(seed_path=None)
        t = ib.add("Can I pay by bank transfer?", customer="Ann", subject="Billing")
        self.assertEqual(ib.counts()["new"], 1)
        ib.set_status(t["id"], "handling")
        ib.set_status(t["id"], "auto-resolved", result={"status": "resolved", "reply": "Yes.",
                                                       "reason_code": "ok", "confidence": 1.0,
                                                       "citation": {"title": "Payment methods"}})
        pub = ib.set_status(t["id"], "sent")
        self.assertEqual(pub["status"], "sent")
        self.assertEqual(pub["outcome"]["citation"], "Payment methods")
        self.assertIsNotNone(pub["sent_at"])
        self.assertEqual(ib.counts()["sent"], 1)

    def test_rejects_unknown_status(self):
        ib = Inbox(seed_path=None)
        t = ib.add("hello")
        with self.assertRaises(ValueError):
            ib.set_status(t["id"], "closed")


class LiveEvents(unittest.TestCase):
    """predict(on_event=...) must emit every step as it happens, in order, including the
    sampling stage — that is what the console streams."""

    def setUp(self):
        self._orig = (llm.complete, llm.answer_or_abstain)
        llm.complete = lambda prompt, **kw: '{"action":"answer","reason":"found"}'
        llm.answer_or_abstain = lambda q, ctx, temperature=0.0: {
            "answerable": True, "answer": "twenty percent", "reply": "It grew 20%.",
            "cite": 1, "confidence": 1.0}
        agent_mod.llm.answer_or_abstain = llm.answer_or_abstain

    def tearDown(self):
        llm.complete, llm.answer_or_abstain = self._orig
        agent_mod.llm.answer_or_abstain = self._orig[1]

    def test_events_match_steps_and_include_sampling(self):
        docs = ["quarterly revenue grew twenty percent", "the cat sat on the mat"]
        seen = []
        p = Agent(docs, k=1, n_samples=3, retriever="tfidf").predict(
            "how much did revenue grow", on_event=seen.append)
        self.assertEqual([s["action"] for s in seen], ["search", "decide", "sampling", "sample"])
        self.assertEqual(seen, p["steps"])              # the trace is exactly what was streamed
        self.assertEqual(seen[-1]["votes"], 3)
        self.assertEqual(p["reply"], "It grew 20%.")


if __name__ == "__main__":
    unittest.main()
