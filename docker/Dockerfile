# The stock serving image with the router swapped. Nothing else changes: same engine, same
# CUDA, same entrypoint, so anything measured against this image is measured against vLLM.
FROM vllm/vllm-openai:v0.31.0

COPY pyproject.toml /opt/par/pyproject.toml
COPY src /opt/par/src
# `--no-deps` on purpose: the image already pins a torch/CUDA pair that resolving `torch`
# again would happily replace.
RUN pip install --no-deps -e /opt/par

# Unset means stock routing - the plugin patches nothing at all, so this image is a drop-in
# replacement whether or not the rule is turned on. Set it to the number of top ranks chosen
# by score; the rest are re-placed onto ranks the head already visits.
ENV PAR_KEEP=""
