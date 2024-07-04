import warnings

import pyrevolve as pr

from .checkpoint_interface import CheckpointInterface
from .runge_kutta_integrator_pyrevolve_classes import (
    RungeKuttaCheckpoint,
    RungeKuttaIntegratorSymbol,
    RungeKuttaForwardOperator,
    RungeKuttaReverseOperator,
)


class PyrevolveCheckpointer(CheckpointInterface):
    """Checkpointer where checkpointing is done via pyRevolve. Most memory efficient version,
    but doesn't support online checkpointing (i.e. unknown number of time steps)."""

    def __init__(self):
        self.revolver_class_type = None

        self._serialized_old_state_symbol = None
        self._serialized_new_state_symbol = None
        self._forward_operator = None
        self._reverse_operator = None
        self.revolver_options = {}

        self.revolver = None

    def setup(self, **kwargs):

        self._serialized_old_state_symbol = RungeKuttaIntegratorSymbol(
            kwargs["array_size"]
        )
        self._serialized_new_state_symbol = RungeKuttaIntegratorSymbol(
            kwargs["array_size"]
        )

        checkpoint = RungeKuttaCheckpoint(
            {
                "serialized_old_state": self._serialized_old_state_symbol,
                "serialized_new_state": self._serialized_new_state_symbol,
            }
        )

        if "revolver_type" in kwargs:
            revolver_type = kwargs["revolver_type"]
        else:
            revolver_type = "Memory"

        self.revolver_class_type = self._setup_revolver_class_type(revolver_type)

        if "revolver_options" in kwargs:
            for key, value in kwargs["revolver_options"].items():
                if revolver_type == "MultiLevel" and key == "storage_list":
                    storage_list = []
                    for storage_type, options in value.items():
                        if storage_type == "Numpy":
                            storage_list.append(
                                pr.NumpyStorage(checkpoint.size, **options)
                            )
                        elif storage_type == "Disk":
                            storage_list.append(
                                pr.DiskStorage(checkpoint.size, **options)
                            )
                        elif storage_type == "Bytes":
                            storage_list.append(
                                pr.BytesStorage(checkpoint.size, **options)
                            )
                    self.revolver_options[key] = storage_list
                else:
                    self.revolver_options[key] = value
        self.revolver_options["checkpoint"] = checkpoint
        self.revolver_options["n_timesteps"] = kwargs["num_steps"]
        if "n_checkpoints" not in self.revolver_options:
            if revolver_type not in ["MultiLevel", "Base"]:
                self.revolver_options["n_checkpoints"] = (
                    1 if kwargs["num_steps"] == 1 else None
                )

        self.revolver_options["fwd_operator"] = RungeKuttaForwardOperator(
            self._serialized_old_state_symbol,
            self._serialized_new_state_symbol,
            kwargs["run_step_func"],
        )
        self.revolver_options["rev_operator"] = RungeKuttaReverseOperator(
            self._serialized_old_state_symbol,
            kwargs["array_size"],
            kwargs["run_step_jacvec_rev_func"],
        )

    def create_checkpointer(self):
        self._revolver = self.revolver_class_type(**self.revolver_options)

    def _setup_revolver_class_type(self, revolver_type: str):
        if revolver_type == "SingleLevel":
            return pr.SingleLevelRevolver
        elif revolver_type == "MultiLevel":
            warnings.warn(
                """MultiLevelRevolver currently has problems where certain numbers of checkpoints work and others don't 
                (without an obvious reason why). Use with care."""
            )
            return pr.MultiLevelRevolver
        elif revolver_type == "Disk":
            return pr.DiskRevolver
        elif revolver_type == "Base":
            return pr.BaseRevolver
        elif revolver_type == "Memory":
            return pr.MemoryRevolver
        else:
            raise TypeError(
                "Given revolver_type is invalid. Options are 'Memory', 'Disk', 'SingleLevel', 'MultiLevel' and Base. "
                f"Given was '{revolver_type}'."
            )

    def iterate_forward(self, initial_state):
        print("starting iterate_forward")
        self._serialized_new_state_symbol.data = initial_state.copy()
        print("data copied")
        self._revolver.apply_forward()
        print("iteration done")

    def iterate_reverse(self, final_state_perturbation):
        self.revolver_options["rev_operator"].serialized_state_perturbations = (
            final_state_perturbation.copy()
        )
        self._revolver.apply_reverse()
