import argparse
import time

import numpy as np
import openmdao.api as om
from openmdao.utils.array_utils import get_evenly_distributed_size

from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.integration_control import IntegrationControl
from rkopenmdao.butcher_tableaux import (
    implicit_euler,
    second_order_two_stage_sdirk,
    third_order_three_stage_sdirk,
    third_order_four_stage_esdirk,
    third_order_five_stage_esdirk,
    fifth_order_six_stage_esdirk,
)


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
            self.par_time = self.options["runtime"] / self.options["core_count"]
            self.par_size = get_evenly_distributed_size(self.comm, self.options["size"])
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
        time.sleep(self.par_time)
        outputs["x_stage"] = self.solution_vector.copy()

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None
    ):
        time.sleep(self.par_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=float)
    parser.add_argument("--core_count", type=int)
    parser.add_argument("--size", type=int)
    parser.add_argument("--scaling_type")

    args = parser.parse_args()

    stage_prob = om.Problem()
    indep = om.IndepVarComp()
    for var in ["x_old", "x_acc_stages"]:
        indep.add_output(var, distributed=True, shape_by_conn=True)
    stage_prob.model.add_subsystem("indep", indep, promotes=["*"])
    stage_prob.model.add_subsystem(
        "par_dummy",
        ParallelDummy(
            runtime=args.runtime,
            core_count=args.core_count,
            size=args.size,
            scaling_type=args.scaling_type,
        ),
        promotes=["*"],
    )
    tableau_dict = {
        "1 stage": implicit_euler,
        "2 stage": second_order_two_stage_sdirk,
        "3 stage": third_order_three_stage_sdirk,
        "4 stage": third_order_four_stage_esdirk,
        "5 stage": third_order_five_stage_esdirk,
        "6 stage": fifth_order_six_stage_esdirk,
    }
    file_name = f"stage_time_{args.runtime}_core_count_{args.core_count}"
    file_name = file_name + f"_size_{args.size}_scaling_{args.scaling_type}.txt"
    with open(file_name, mode="w") as f:
        for stage_num_string, tableau in tableau_dict.items():
            integration_control = IntegrationControl(0.0, 20, 0.1)
            rk_prob = om.Problem()
            rk_indep = om.IndepVarComp()
            rk_indep.add_output("x_initial", distributed=True, shape_by_conn=True)
            rk_prob.model.add_subsystem("rk_indep", rk_indep, promotes=["*"])
            rk_prob.model.add_subsystem(
                "rk_integration",
                RungeKuttaIntegrator(
                    time_stage_problem=stage_prob,
                    time_integration_quantities=["x"],
                    integration_control=integration_control,
                    butcher_tableau=tableau,
                ),
                promotes=["*"],
            )
            rk_prob.setup()
            rk_prob["x_initial"][:].fill(0.0)

            t1 = time.perf_counter()
            rk_prob.run_model()
            t2 = time.perf_counter()
            f.write(f"{stage_num_string} {t2-t1}\n")
