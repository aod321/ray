import abc
import time
from typing import Dict, Any, Optional

from anyscale.sdk.anyscale_client.sdk import AnyscaleSDK

from ray_release.util import dict_hash, get_anyscale_sdk, anyscale_cluster_url


class ClusterManager(abc.ABC):
    def __init__(self,
                 test_name: str,
                 project_id: str,
                 sdk: Optional[AnyscaleSDK] = None):
        self.sdk = sdk or get_anyscale_sdk()

        self.test_name = test_name
        self.project_id = project_id

        self.cluster_name = f"{test_name}_{int(time.time())}"
        self.cluster_id = None

        self.cluster_env = None
        self.cluster_env_name = None
        self.cluster_env_id = None
        self.cluster_env_build_id = None

        self.cluster_compute = None
        self.cluster_compute_name = None
        self.cluster_compute_id = None

        self.autosuspend_minutes = 30  # Todo: set to 10 for prod

    def set_cluster_env(self, cluster_env: Dict[str, Any]):
        self.cluster_env = cluster_env
        self.cluster_env_name = (f"{self.project_id}__env__{self.test_name}__"
                                 f"{dict_hash(cluster_env)}")

    def set_cluster_compute(self, cluster_compute: Dict[str, Any]):
        self.cluster_compute = cluster_compute
        self.cluster_compute_name = (
            f"{self.project_id}__compute__{self.test_name}__"
            f"{dict_hash(cluster_compute)}")

    def build_configs(self, timeout: float = 30.):
        raise NotImplementedError

    def delete_configs(self):
        raise NotImplementedError

    def start_cluster(self, timeout: float = 600.):
        raise NotImplementedError

    def terminate_cluster(self):
        raise NotImplementedError

    def get_cluster_address(self) -> str:
        raise NotImplementedError

    def get_cluster_url(self) -> Optional[str]:
        if not self.project_id or not self.cluster_id:
            return None
        return anyscale_cluster_url(self.project_id, self.cluster_id)
