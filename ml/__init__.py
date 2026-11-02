"""Machine-learning package: text preprocessing, training and explainability."""

LABELS: dict[int, str] = {0: "Fake News", 1: "Real News"}
"""Class index -> human readable label. 0 = fake, 1 = real (matches the original notebook)."""

FAKE, REAL = 0, 1
