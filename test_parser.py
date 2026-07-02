import json
import os
from unittest.mock import MagicMock, patch

import pytest

import db as db_module
import parser

TEST_CV_AHMET = os.path.join(os.path.dirname(__file__), "test_cv_1_ahmet.docx")
TEST_CV_ZEYNEP = os.path.join(os.path.dirname(__file__), "test_cv_2_zeynep.docx")


class TestExtractTextFromDocx:
    def test_extracts_nonempty_text_from_ahmet_cv(self):
        text = parser.extract_text_from_docx(TEST_CV_AHMET)
        assert isinstance(text, str)
        assert text.strip() != ""

    def test_extracts_nonempty_text_from_zeynep_cv(self):
        text = parser.extract_text_from_docx(TEST_CV_ZEYNEP)
        assert isinstance(text, str)
        assert text.strip() != ""


class TestParseCvWithGemini:
    def test_parses_cv_text_into_schema_dict(self):
        fake_cv_data = {
            "ad_soyad": "Ahmet Yilmaz",
            "iletisim": {"e_posta": "ahmet@example.com", "telefon": "5551112233"},
            "deneyim_yili": 5,
            "yetenekler": ["Python", "SQL"],
            "egitim": ["Bilgisayar Muhendisligi"],
            "ozet": "Deneyimli yazilim gelistirici.",
        }
        fake_response = MagicMock()
        fake_response.text = json.dumps(fake_cv_data)

        with patch.object(parser.client.models, "generate_content", return_value=fake_response) as mock_call:
            result = parser.parse_cv_with_gemini("herhangi bir CV metni")

        mock_call.assert_called_once()
        assert result == fake_cv_data


class TestGetEmbedding:
    def test_returns_vector_from_gemini_response(self):
        fake_values = [0.1, 0.2, 0.3]
        fake_embedding = MagicMock()
        fake_embedding.values = fake_values
        fake_response = MagicMock()
        fake_response.embeddings = [fake_embedding]

        with patch.object(parser.client.models, "embed_content", return_value=fake_response) as mock_call:
            result = parser.get_embedding("herhangi bir metin")

        mock_call.assert_called_once()
        assert result == fake_values


class TestCalculateCosineSimilarity:
    def test_identical_vectors_score_100(self):
        v = [1.0, 2.0, 3.0]
        assert parser.calculate_cosine_similarity(v, v) == pytest.approx(100.0)

    def test_orthogonal_vectors_score_0(self):
        assert parser.calculate_cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_returns_0(self):
        assert parser.calculate_cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestEndToEndMatchingMocked:
    def _mock_embedding_response(self, values):
        fake_embedding = MagicMock()
        fake_embedding.values = values
        fake_response = MagicMock()
        fake_response.embeddings = [fake_embedding]
        return fake_response

    def test_matching_score_between_0_and_100_for_each_test_cv(self):
        job_vector = [1.0, 0.0, 0.0]

        for cv_path, cv_vector_values in [
            (TEST_CV_AHMET, [0.9, 0.1, 0.0]),
            (TEST_CV_ZEYNEP, [0.2, 0.8, 0.0]),
        ]:
            cv_text = parser.extract_text_from_docx(cv_path)
            assert cv_text.strip() != ""

            with patch.object(
                parser.client.models,
                "embed_content",
                return_value=self._mock_embedding_response(cv_vector_values),
            ):
                cv_vector = parser.get_embedding(cv_text)

            score = parser.calculate_cosine_similarity(job_vector, cv_vector)
            assert 0.0 <= score <= 100.0


class TestDb:
    def test_save_and_get_embedding_roundtrip(self):
        db_module.init_db()

        row_id = db_module.save_embedding("cv", "test_cv_1_ahmet.docx", "ornek metin", [0.1, 0.2, 0.3])
        try:
            vector = db_module.get_embedding(row_id)
            assert vector == pytest.approx([0.1, 0.2, 0.3])
        finally:
            db_module.delete_embedding(row_id)
