import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def captioner():
    from lcr.captioner import Captioner
    return Captioner()


@pytest.fixture(scope="session")
def sample_images():
    from lcr import data
    # Development tests never need to reopen validation images.
    return [data.load_image(i) for i in data.ids("B")[:4]]
