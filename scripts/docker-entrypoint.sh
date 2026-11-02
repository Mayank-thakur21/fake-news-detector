#!/bin/sh
# Train models on first boot when none are present, then exec the main command.
set -eu

MODEL_DIR="${MODEL_DIR:-/app/models}"

if [ ! -f "$MODEL_DIR/vectorizer.joblib" ]; then
    if [ "${AUTO_TRAIN:-true}" = "true" ]; then
        echo "[entrypoint] No models in $MODEL_DIR - training now (this runs once)..."
        # Uses data/Fake.csv + data/True.csv when mounted, else the bundled demo sample.
        python -m ml.train --model-dir "$MODEL_DIR" ${TRAIN_ARGS:-}
    else
        echo "[entrypoint] WARNING: no models found and AUTO_TRAIN=false; /predict will return 503."
    fi
fi

exec "$@"
