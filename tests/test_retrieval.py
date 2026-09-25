import unittest

from jev_rag_retrieval.retrieval import retrieve, split_passages


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


if __name__ == "__main__":
    unittest.main()
