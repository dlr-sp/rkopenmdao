import openmdao.api as om
import numpy as np
from rkopenmdao.integration_control import (
    IntegrationControl,
    StepTerminationIntegrationControl,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator


class ParallelDummy(om.ExplicitComponent):
    """Component that emulates a piece of parallelized code. Given a sequential runtime
    r  (in seconds) and a core count p, runs as long as a perfectly parallelized program
    would, i.e r/p seconds. Expects to be run on p processes. Also accepts a problem
    size s. Depending on the scalinf type (weak/strong), s is divided or multiplied by
    p."""

    par_time: float
    par_size: int
    solution_vector: np.ndarray

    def initialize(self):
        self.options.declare(
            "runtime", types=float, desc="Sequential runtime in seconds"
        )
        self.options.declare("core_count", types=int, desc="Number of cores")
        self.options.declare("size", types=int, desc="Base size of the problem")
        self.options.declare(
            "scaling_type",
            values=["weak", "strong"],
            desc="What kind of scaling is to be investigated.",
        )

    def setup(self):
        if self.options["scaling_type"] == "weak":
            self.par_time = self.options["runtime"]
            self.par_size = self.options["size"]
        else:
            print("Not implemented yet")
        self.solution_vector = np.ones(self.par_size)

        self.add_input(
            "x_old",
            tags=["step_input_var", "x"],
            distributed=True,
            val=np.zeros(self.par_size),
        )
        self.add_input(
            "x_acc_stages",
            tags=["accumulated_stage_var", "x"],
            distributed=True,
            val=np.zeros(self.par_size),
        )
        self.add_output(
            "x_stage",
            tags=["stage_output_var", "x"],
            distributed=True,
            val=np.zeros(self.par_size),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # time.sleep(self.par_time)
        outputs["x_stage"] = self.solution_vector.copy()

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None
    ):
        # time.sleep(self.par_time)
        """NO"""