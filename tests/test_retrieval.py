import unittest
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from jev_rag_retrieval.retrieval import _jev_score, from_sivtr_workset, redact_for_remote, retrieve, split_passages


class RetrievalTests(unittest.TestCase):
    def test_two_stage_ranking_preserves_refs_and_session_boundaries(self):
        records = [
            {"session_id": "a", "workspace": "one", "workref": "@a1", "text": "The parser benchmark failed on CSV imports."},
            {"session_id": "a", "workspace": "one", "workref": "@a2", "text": "Fixed delimiter detection and verified the regression test."},
            {"session_id": "b", "workspace": "two", "workref": "@b1", "text": "Parser planning discussion without a test result."},
        ]

        def scorer(query, candidate, stage):
            return 0.95 if ("verified" in candidate or "regression" in candidate) else 0.12

        result = retrieve("Which parser regression was fixed?", records, scorer=scorer, use_jev=True)
        self.assertEqual(result["sessions"][0]["session_id"], "a")
        self.assertEqual(result["sessions"][0]["passages"][0]["workref"], "@a2")
        self.assertEqual(result["mode"], "jev")

    def test_chunking_keeps_original_ref_and_offsets(self):
        parts = split_passages({"session_id": "s", "workref": "@r", "text": "a" * 3000})
        self.assertGreater(len(parts), 2)
        self.assertTrue(all(part["workref"] == "@r" for part in parts))
        self.assertEqual(parts[0]["char_start"], 0)

    def test_api_errors_fall_back_to_lexical(self):
        def broken(*_):
            raise RuntimeError("sensitive exception content must not be shown")

        result = retrieve("parser", [{"session_id": "s", "workref": "@r", "text": "parser fixed"}], scorer=broken, use_jev=True)
        self.assertEqual(result["mode"], "lexical_fallback")
        self.assertNotIn("sensitive exception content", str(result))

    def test_cannot_add_missing_candidate(self):
        result = retrieve("unknown decision", [{"session_id": "s", "workref": "@r", "text": "other work"}], use_jev=False)
        self.assertEqual(result["input_sessions"], 1)
        self.assertEqual(result["sessions"][0]["session_id"], "s")

    def test_remote_text_redacts_credentials(self):
        text = "This session contains apikey_abcdefghijklmnopqrstuvwxyz_1234567890 and Bearer abcdefghijklmnopqrstuvwxyz123456."
        cleaned = redact_for_remote(text)
        self.assertNotIn("apikey_abcdefghijklmnopqrstuvwxyz", cleaned)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz123456", cleaned)
        self.assertIn("[REDACTED_CREDENTIAL]", cleaned)

    def test_sivtr_workset_conversion_keeps_record_ref(self):
        source = {"cwd": "/project", "records": [{
            "work_ref": "codex/session/3", "session": {"canonical_id": "session"},
            "title": "Parser fix", "time": {"started_at": "2026-09-25T10:00:00Z"},
            "parts": [
                {"kind": "message", "role": "user", "content": {"Text": {"content": "Why did parser tests fail?"}}},
                {"kind": "action", "role": None, "content": None},
                {"kind": "message", "role": "assistant", "content": {"Text": {"content": "Fixed delimiter detection."}}},
            ],
        }]}
        records = from_sivtr_workset(source)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["workref"], "codex/session/3")
        self.assertIn("Fixed delimiter", records[0]["text"])

    def test_jev_request_redacts_configured_key(self):
        seen = {}
        sdk = ModuleType("typesafe_sdk")

        class Client:
            def __init__(self, **_):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def system_one(self, *, state, questions):
                seen.update(state)
                return SimpleNamespace(answers={"relevant": SimpleNamespace(noul=0.8)})

        sdk.TypeSafeClient = Client
        sdk.Noul = lambda **_: object()
        sdk.NoulCriteria = lambda **_: object()
        dummy = "apikey_" + "x" * 50
        with patch.dict(sys.modules, {"typesafe_sdk": sdk}), patch.dict(os.environ, {"TYPESAFE_API_KEY": dummy}):
            score = _jev_score("Find a fix", f"The key is {dummy}; the parser was fixed.", "passage")
        self.assertEqual(score, 0.8)
        self.assertNotIn(dummy, str(seen))
        self.assertIn("parser was fixed", seen["candidate"])


if __name__ == "__main__":
    unittest.main()
