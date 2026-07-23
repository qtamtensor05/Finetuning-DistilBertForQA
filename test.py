from __future__ import annotations

import csv
import json
from pathlib import Path

from ..utils.schemas import Document


TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".py", ".yaml", ".yml", ".csv", ".json", ".jsonl"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".docx"}