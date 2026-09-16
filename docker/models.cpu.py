# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""CPU deployment config: an embedding model plus the framework-free example detector.

Used by the CPU image variant (``TORCH_INDEX_URL`` pointing at the CPU wheels,
``INSTALL_CLIP=true``, ``INSTALL_EXAMPLE=true``). It serves real embeddings without a GPU,
which is what makes an end-to-end test possible on a developer machine.

MobileCLIP2-S2 is small enough to run on CPU at a usable pace; weights are fetched once on
first load and cached in the persistent volume.
"""

from pixano_inference.configs import DeploymentConfig, ModelConfig


models = [
    ModelConfig(
        name="clip",
        model_class="OpenClipEmbeddingModel",
        model_params={"path": "MobileCLIP2-S2", "pretrained": "dfndr2b", "compile": False},
        deployment=DeploymentConfig(num_gpus=0, num_cpus=2, min_replicas=1, max_replicas=1),
    ),
    ModelConfig(
        name="numpy-detector",
        model_class="NumpyDetector",
        model_params={"threshold": 20},
        deployment=DeploymentConfig(num_gpus=0, num_cpus=1, min_replicas=1, max_replicas=1),
    ),
]
