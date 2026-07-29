# -*- coding: utf-8 -*-
"""Offline contract tests for financial RAG metadata and retrieval provenance.

The tests deliberately mock ChromaDB: they document the public API contract
without downloading an embedding model or touching the persistent vector store.
"""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The contract tests mock all vector-store interaction.  A minimal import stub
# keeps them runnable in lightweight CI environments without chromadb installed.
try:
    import chromadb  # noqa: F401
except ModuleNotFoundError:
    chromadb_stub = types.ModuleType("chromadb")
    chromadb_stub.PersistentClient = object
    chromadb_stub.utils = types.SimpleNamespace(
        embedding_functions=types.SimpleNamespace()
    )
    sys.modules["chromadb"] = chromadb_stub


class FakeCollection:
    """Small Chroma collection double that records add/query calls."""

    def __init__(self, query_result=None):
        self.add_calls = []
        self.query_calls = []
        self.query_result = query_result or {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
        }

    def add(self, **kwargs):
        self.add_calls.append(kwargs)

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return self.query_result

    def get(self, **_kwargs):
        return {"metadatas": []}


class TestFinancialRagRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend import accounting_rag
        cls.rag = accounting_rag

    def test_upload_persists_financial_metadata_on_every_chunk(self):
        """Document-level provenance must remain available after chunking."""
        collection = FakeCollection()
        metadata = {
            "ticker": "600519",
            "industry": "spirits",
            "document_type": "annual_report",
            "source_url": "https://example.test/600519-2025-ar.pdf",
            "published_at": "2026-03-30",
            "effective_date": "2026-03-30",
            "expires_at": "2027-03-30",
            "trust_level": "exchange_filing",
        }

        with tempfile.TemporaryDirectory() as upload_dir, \
             patch.object(self.rag, "_ensure_init"), \
             patch.object(self.rag, "_UPLOAD_DIR", upload_dir), \
             patch.object(self.rag, "_parse_file", return_value="body"), \
             patch.object(self.rag, "_chunk_text", return_value=["chunk one", "chunk two"]), \
             patch.object(self.rag, "_get_kb_collection", return_value=collection):
            result = self.rag.upload_document(
                b"raw document", "moutai.pdf", "industry", metadata=metadata
            )

        self.assertEqual(result["kb"], "industry")
        self.assertEqual(len(collection.add_calls), 1)
        stored = collection.add_calls[0]["metadatas"]
        self.assertEqual(len(stored), 2)
        for index, chunk_metadata in enumerate(stored):
            self.assertEqual(chunk_metadata["chunk_index"], index)
            self.assertEqual(chunk_metadata["filename"], "moutai.pdf")
            # ticker is accepted as an upload convenience alias, but the
            # stored/filterable schema has one canonical symbol field.
            self.assertEqual(chunk_metadata["symbol"], "600519")
            for key in ("industry", "document_type", "source_url", "published_at",
                        "effective_date", "expires_at", "trust_level"):
                self.assertEqual(chunk_metadata[key], metadata[key])

    def test_search_converts_multi_field_filter_to_chroma_and(self):
        """Independent filters must be combined rather than one silently winning."""
        collection = FakeCollection()
        metadata_filter = {"ticker": "600519", "document_type": "annual_report"}

        with patch.object(self.rag, "_ensure_init"), \
             patch.object(self.rag, "_get_kb_collection", return_value=collection):
            self.rag.search_kb(
                "moutai revenue", "industry", top_k=7, metadata_filter=metadata_filter
            )

        query = collection.query_calls[0]
        self.assertGreaterEqual(query["n_results"], 7)
        self.assertEqual(
            query["where"],
            {"$and": [{"symbol": "600519"}, {"document_type": "annual_report"}]},
        )

    def test_search_keeps_single_filter_shape_and_discards_distant_hits(self):
        """Distance threshold is applied after retrieval, without changing a simple filter."""
        collection = FakeCollection({
            "ids": [["doc_a_chunk_0", "doc_b_chunk_0"]],
            "documents": [["near evidence", "distant noise"]],
            "metadatas": [[
                {"doc_id": "doc_a", "filename": "a.pdf", "chunk_index": 0,
                 "total_chunks": 1, "symbol": "600519"},
                {"doc_id": "doc_b", "filename": "b.pdf", "chunk_index": 0,
                 "total_chunks": 1, "symbol": "600519"},
            ]],
            "distances": [[0.12, 0.88]],
        })

        with patch.object(self.rag, "_ensure_init"), \
             patch.object(self.rag, "_get_kb_collection", return_value=collection):
            result = self.rag.search_kb(
                "600519", metadata_filter={"ticker": "600519"}, max_distance=0.5
            )

        self.assertEqual(collection.query_calls[0]["where"], {"symbol": "600519"})
        self.assertEqual([item["doc_id"] for item in result], ["doc_a"])
        self.assertEqual(result[0]["distance"], 0.12)

    def test_search_returns_auditable_provenance_fields(self):
        """A generated answer can cite a retrieved chunk without guessing its origin."""
        collection = FakeCollection({
            "ids": [["ar_2025_chunk_3"]],
            "documents": [["Revenue grew 15% year over year."]],
            "metadatas": [[{
                "doc_id": "ar_2025",
                "filename": "600519-2025-annual-report.pdf",
                "chunk_index": 3,
                "total_chunks": 9,
                "source_url": "https://example.test/600519-2025-ar.pdf",
                "published_at": "2026-03-30",
                "symbol": "600519",
                "document_type": "annual_report",
            }]],
            "distances": [[0.2]],
        })

        with patch.object(self.rag, "_ensure_init"), \
             patch.object(self.rag, "_get_kb_collection", return_value=collection):
            item = self.rag.search_kb("revenue", "industry", hybrid=False)[0]

        self.assertEqual(item["content"], "Revenue grew 15% year over year.")
        self.assertEqual(item["doc_id"], "ar_2025")
        self.assertEqual(item["chunk_id"], "ar_2025_chunk_3")
        self.assertEqual(item["source_url"], "https://example.test/600519-2025-ar.pdf")
        self.assertEqual(item["published_at"], "2026-03-30")
        self.assertEqual(item["metadata"]["symbol"], "600519")
        self.assertTrue(item["citation_id"])
        self.assertTrue(item["citation"])
        self.assertEqual(item["citation"]["filename"], "600519-2025-annual-report.pdf")
        self.assertEqual(item["citation"]["chunk_id"], "ar_2025_chunk_3")
        self.assertIn("vector_score", item)
        self.assertIn("keyword_score", item)


if __name__ == "__main__":
    unittest.main()
