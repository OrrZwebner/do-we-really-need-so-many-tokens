import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nelson_rag.config import Settings  # noqa: E402
from nelson_rag.indexing.build import build_index  # noqa: E402
from nelson_rag.session import open_session  # noqa: E402

FIXTURE_SOURCE = Path(__file__).parent / "fixtures" / "source"


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    """Settings pointed at a throwaway index built with the offline backend."""
    index_dir = tmp_path_factory.mktemp("index")
    return Settings(
        index_dir=index_dir,
        source_dir=FIXTURE_SOURCE,
        book_edition="synthetic test fixture",
        embedding_backend="hash",
        chunk_chars=800,
        chunk_overlap_chars=120,
        min_chunk_chars=80,
    )


@pytest.fixture(scope="session")
def built_index(settings):
    return build_index(settings)


@pytest.fixture(scope="session")
def session(settings, built_index):
    s = open_session(settings)
    yield s
    s.close()
