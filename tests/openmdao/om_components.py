import numpy as np
from rkopenmdao.components import ExplicitUnsteadyComponent, ImplicitUnsteadyComponent


class ODE1dParameter(ExplicitUnsteadyComponent):
    """
    Models the stage of the ODE x' = a * x. The following formula for the stage
    results:
    k_i = b * (x_n + dt * s_i)/(1 - dt * a_ii)
    The formulas for the fwd/rev derivatives are:
    (fwd) dk_i = (dx_n + dt * ds_i)/(1 - dt * a_ii)
    (rev) dx_n = dk_i / (1 - dt * a_ii)
    (rev) ds_i = dt * dk_i / (1 - dt * a_ii)

    This is the simplest linear ODE (apart from x' = const), so this is the least that
    has to work.
    """

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_input("b", val=1, shape=1, tags=["time_independent_input_var", "b"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        outputs["x_stage"] = (
            inputs["b"]
            * (inputs["x"] + self.om_data_exchange.step_size * inputs["acc_stages"])
            / (1 - self.om_data_exchange.step_size * self.om_data_exchange.stage_factor)
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        divisor = (
            1 - self.om_data_exchange.step_size * self.om_data_exchange.stage_factor
        )
        if mode == "fwd":
            d_outputs["x_stage"] += inputs["b"] * d_inputs["x"] / divisor
            d_outputs["x_stage"] += (
                inputs["b"]
                * self.om_data_exchange.step_size
                * d_inputs["acc_stages"]
                / divisor
            )
            d_outputs["x_stage"] += (
                (inputs["x"] + self.om_data_exchange.step_size * inputs["acc_stages"])
                / divisor
                * d_inputs["b"]
            )
        elif mode == "rev":
            d_inputs["x"] += inputs["b"] * d_outputs["x_stage"] / divisor
            d_inputs["acc_stages"] += (
                inputs["b"]
                * self.om_data_exchange.step_size
                * d_outputs["x_stage"]
                / divisor
            )
            d_inputs["b"] += (
                (inputs["x"] + self.om_data_exchange.step_size * inputs["acc_stages"])
                / divisor
                * d_outputs["x_stage"]
            )


def ode1d_analytical_solution(time, initial_value, initial_time, param=1.0):
    """Analytical solution to the ODE of the above component."""
    return initial_value * np.exp(param * (time - initial_time))


class ODE2dUnified(ExplicitUnsteadyComponent):
    """
    Models the stage of the ODE system  x' = y, y' = x. The following formula for the
    stage
    results:
    k_i^1 = (x_n^2 + dt * s_i^2 + dt * a_ii * (x_n^1 + dt * s_i^1))
            / (1-dt**2 * a_ii**2)
    k_i^2 = (x_n^1 + dt * s_i^1 + dt * a_ii * (x_n^2 + dt * s_i^2))
            / (1-dt**2 * a_ii**2)
    The formulas for the fwd/rev derivatives are:
    (fwd)
    dk_i^1 = (dx_n^2 + dt * ds_i^2 + dt * a_ii * (dx_n^1 + dt * ds_i^1))
             / (1-dt**2 * a_ii**2)
    dk_i^2 = (dx_n^1 + dt * ds_i^1 + dt * a_ii * (dx_n^2 + dt * ds_i^2))
             /(1-dt**2 * a_ii**2)
    (rev)
    dx_n^1 = ((dt * a_ii) * dk_i^1 + dk_i^2)/(1-dt**2 * a_ii**2)
    ds_i^1 = dt * ((dt * a_ii) * dk_i^1 + dk_i^2)/(1-dt**2 * a_ii**2)
    dx_n^2 = ((dt * a_ii) * dk_i^2 + dk_i^1)/(1-dt**2 * a_ii**2)
    ds_i^2 = dt * ((dt * a_ii) * dk_i^2 + dk_i^1)/(1-dt**2 * a_ii**2)
    """

    def setup(self):
        self.add_input("x", shape=2, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=2, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=2, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        factor = self.om_data_exchange.step_size * self.om_data_exchange.stage_factor
        outputs["x_stage"][0] = (
            factor * inputs["x"][0]
            + inputs["x"][1]
            + self.om_data_exchange.step_size
            * (factor * inputs["acc_stages"][0] + inputs["acc_stages"][1])
        ) / (1 - factor**2)
        outputs["x_stage"][1] = (
            inputs["x"][0]
            + factor * inputs["x"][1]
            + self.om_data_exchange.step_size
            * (inputs["acc_stages"][0] + factor * inputs["acc_stages"][1])
        ) / (1 - factor**2)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        factor = self.om_data_exchange.step_size * self.om_data_exchange.stage_factor

        if mode == "fwd":
            d_outputs["x_stage"][0] += (
                factor * d_inputs["x"][0] + d_inputs["x"][1]
            ) / (1 - factor**2)
            d_outputs["x_stage"][1] += (
                d_inputs["x"][0] + factor * d_inputs["x"][1]
            ) / (1 - factor**2)
            d_outputs["x_stage"][0] += (
                self.om_data_exchange.step_size
                * (factor * d_inputs["acc_stages"][0] + d_inputs["acc_stages"][1])
                / (1 - factor**2)
            )
            d_outputs["x_stage"][1] += (
                self.om_data_exchange.step_size
                * (d_inputs["acc_stages"][0] + factor * d_inputs["acc_stages"][1])
                / (1 - factor**2)
            )
        elif mode == "rev":
            d_inputs["x"][0] += (
                factor * d_outputs["x_stage"][0] + d_outputs["x_stage"][1]
            ) / (1 - factor**2)
            d_inputs["x"][1] += (
                d_outputs["x_stage"][0] + factor * d_outputs["x_stage"][1]
            ) / (1 - factor**2)
            d_inputs["acc_stages"][0] += (
                self.om_data_exchange.step_size
                * (factor * d_outputs["x_stage"][0] + d_outputs["x_stage"][1])
                / (1 - factor**2)
            )
            d_inputs["acc_stages"][1] += (
                self.om_data_exchange.step_size
                * (d_outputs["x_stage"][0] + factor * d_outputs["x_stage"][1])
                / (1 - factor**2)
            )


# The next two components also model the ODE system  x' = y, y'=x. However, this time
# the formulas are implemented in 2 components. This can then be used to test that both
# ways (one or two components) work the same.


class ODE2dSplit1(ExplicitUnsteadyComponent):
    """
    Models the stage of the ODE x' = y. The following formula for the stage
    results:
    k_i^1 = x_n^2 + dt * s_i^2 + dt * a_ii * k_i^2
    The formulas for the fwd/rev derivatives are:
    (fwd) dk_i^1 = dx_n^2 + dt * ds_i^2 + dt * a_ii * (dk_i^2)
    (rev) dx_n^2 = dk_i^1
    (rev) ds_i^2 = dt * dk_i^1
    (rev) dk_i^2 = dt * a_ii * dk_i^1
    """

    def setup(self):
        self.add_input("y", shape=1, tags=["step_input_var", "y"])
        self.add_input("acc_stages_y", shape=1, tags=["accumulated_stage_var", "y"])
        self.add_input("y_stage", shape=1)
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        outputs["x_stage"] = (
            inputs["y"]
            + self.om_data_exchange.step_size * inputs["acc_stages_y"]
            + self.om_data_exchange.step_size
            * self.om_data_exchange.stage_factor
            * inputs["y_stage"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            d_outputs["x_stage"] += d_inputs["y"]
            d_outputs["x_stage"] += (
                self.om_data_exchange.step_size * d_inputs["acc_stages_y"]
            )
            d_outputs["x_stage"] += (
                self.om_data_exchange.step_size
                * self.om_data_exchange.stage_factor
                * d_inputs["y_stage"]
            )
        elif mode == "rev":
            d_inputs["y"] += d_outputs["x_stage"]
            d_inputs["acc_stages_y"] += (
                self.om_data_exchange.step_size * d_outputs["x_stage"]
            )
            d_inputs["y_stage"] += (
                self.om_data_exchange.step_size
                * self.om_data_exchange.stage_factor
                * d_outputs["x_stage"]
            )


class ODE2dSplit2(ExplicitUnsteadyComponent):
    """
    Models the stage of the ODE y' = x. The following formula for the stage
    results:
    k_i^2 = x_n^1 + dt * s_i^1 + dt * a_ii * k_i^1
    The formulas for the fwd/rev derivatives are:
    (fwd) dk_i^2 = dx_n^1 + dt * ds_i^1 + dt * a_ii * (dk_i^1)
    (rev) dx_n^1 = dk_i^2
    (rev) ds_i^1 = dt * dk_i^2
    (rev) dk_i^1 = dt * a_ii * dk_i^2
    """

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages_x", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_input("x_stage", shape=1)
        self.add_output("y_stage", shape=1, tags=["stage_output_var", "y"])

    def compute(self, inputs, outputs):
        outputs["y_stage"] = (
            inputs["x"]
            + self.om_data_exchange.step_size * inputs["acc_stages_x"]
            + self.om_data_exchange.step_size
            * self.om_data_exchange.stage_factor
            * inputs["x_stage"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            d_outputs["y_stage"] += d_inputs["x"]
            d_outputs["y_stage"] += (
                self.om_data_exchange.step_size * d_inputs["acc_stages_x"]
            )
            d_outputs["y_stage"] += (
                self.om_data_exchange.step_size
                * self.om_data_exchange.stage_factor
                * d_inputs["x_stage"]
            )
        elif mode == "rev":
            d_inputs["x"] += d_outputs["y_stage"]
            d_inputs["acc_stages_x"] += (
                self.om_data_exchange.step_size * d_outputs["y_stage"]
            )
            d_inputs["x_stage"] += (
                self.om_data_exchange.step_size
                * self.om_data_exchange.stage_factor
                * d_outputs["y_stage"]
            )


def ode2d_analytical_solution(time, initial_value, initial_time):
    """Analytical solution to the ODE of the above components."""
    return np.array(
        [
            initial_value[0] * np.cosh(time - initial_time)
            + initial_value[1] * np.sinh(time - initial_time),
            initial_value[0] * np.sinh(time - initial_time)
            + initial_value[1] * np.cosh(time - initial_time),
        ]
    )


# The following two components model the system
#   x_1' = x_4
#   x_2' = x_1
#   x_3' = x_2
#   x_4' = x_3
class ODE4dDistributedSplit1(ImplicitUnsteadyComponent):
    """Models the first two equations from above, with the first being on rank 0 and the
    second on rank 1"""

    def setup(self):
        self.add_input("x43", shape=1, distributed=True)
        self.add_input(
            "x12_old", shape=1, distributed=True, tags=["x12", "step_input_var"]
        )
        self.add_input(
            "s12_i", shape=1, distributed=True, tags=["x12", "accumulated_stage_var"]
        )
        self.add_output(
            "k12_i", shape=1, distributed=True, tags=["x12", "stage_output_var"]
        )
        self.add_output("x12", shape=1, distributed=True)

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        delta_t = self.om_data_exchange.step_size
        residuals["x12"] = (
            inputs["x12_old"]
            + delta_t * (inputs["s12_i"] + butcher_diagonal_element * outputs["k12_i"])
            - outputs["x12"]
        )
        exch_data = np.zeros(1)
        if self.comm.rank == 0:
            exch_data[0] = outputs["x12"][0]
            self.comm.Send(exch_data, dest=1, tag=0)
            residuals["k12_i"] = inputs["x43"] - outputs["k12_i"]
        elif self.comm.rank == 1:
            self.comm.Recv(exch_data, source=0, tag=0)
            residuals["k12_i"] = exch_data[0] - outputs["k12_i"]

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        delta_t = self.om_data_exchange.step_size
        if mode == "fwd":
            d_residuals["x12"] += (
                d_inputs["x12_old"]
                + delta_t
                * (d_inputs["s12_i"] + butcher_diagonal_element * d_outputs["k12_i"])
                - d_outputs["x12"]
            )
            exch_data = np.zeros(1)
            if self.comm.rank == 0:
                exch_data[0] = d_outputs["x12"][0]
                self.comm.Send(exch_data, dest=1, tag=0)
                d_residuals["k12_i"] += d_inputs["x43"] - d_outputs["k12_i"]

            elif self.comm.rank == 1:
                self.comm.Recv(exch_data, source=0, tag=0)
                d_residuals["k12_i"] += exch_data[0] - d_outputs["k12_i"]
        # but they seem wrong somehow?
        elif mode == "rev":
            d_inputs["x12_old"] += d_residuals["x12"]
            d_inputs["s12_i"] += delta_t * d_residuals["x12"]
            d_outputs["k12_i"] += (
                delta_t * butcher_diagonal_element * d_residuals["x12"]
            )
            d_outputs["x12"] -= d_residuals["x12"]
            exch_data = np.zeros(1)
            if self.comm.rank == 0:
                self.comm.Recv(exch_data, source=1, tag=1)
                d_outputs["x12"] += exch_data[0]
                d_inputs["x43"] += d_residuals["k12_i"]
                d_outputs["k12_i"] -= d_residuals["k12_i"]
            elif self.comm.rank == 1:
                exch_data[0] = d_residuals["k12_i"][0]
                self.comm.Send(exch_data, dest=0, tag=1)
                d_outputs["k12_i"] -= d_residuals["k12_i"]


class ODE4dDistributedSplit2(ImplicitUnsteadyComponent):
    """Models the last two equations from above, with the first being on rank 1 and the
    second on rank 0"""

    def setup(self):
        self.add_input("x12", shape=1, distributed=True)
        self.add_input(
            "x43_old", shape=1, distributed=True, tags=["x43", "step_input_var"]
        )
        self.add_input(
            "s43_i", shape=1, distributed=True, tags=["x43", "accumulated_stage_var"]
        )
        self.add_output(
            "k43_i", shape=1, distributed=True, tags=["x43", "stage_output_var"]
        )
        self.add_output("x43", shape=1, distributed=True)

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        delta_t = self.om_data_exchange.step_size
        residuals["x43"] = (
            inputs["x43_old"]
            + delta_t * (inputs["s43_i"] + butcher_diagonal_element * outputs["k43_i"])
            - outputs["x43"]
        )
        exch_data = np.zeros(1)
        if self.comm.rank == 0:
            self.comm.Recv(exch_data, source=1, tag=3)
            residuals["k43_i"] = exch_data[0] - outputs["k43_i"]
        elif self.comm.rank == 1:
            exch_data[0] = outputs["x43"][0]
            self.comm.Send(exch_data, dest=0, tag=3)
            residuals["k43_i"] = inputs["x12"] - outputs["k43_i"]

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        delta_t = self.om_data_exchange.step_size
        if mode == "fwd":
            d_residuals["x43"] += (
                d_inputs["x43_old"]
                + delta_t
                * (d_inputs["s43_i"] + butcher_diagonal_element * d_outputs["k43_i"])
                - d_outputs["x43"]
            )
            exch_data = np.zeros(1)
            if self.comm.rank == 0:
                self.comm.Recv(exch_data, source=1, tag=3)
                d_residuals["k43_i"] += exch_data[0] - d_outputs["k43_i"]
            elif self.comm.rank == 1:
                exch_data[0] = d_outputs["x43"][0]
                self.comm.Send(exch_data, dest=0, tag=3)
                d_residuals["k43_i"] += d_inputs["x12"] - d_outputs["k43_i"]
        # but they seem wrong somehow?
        elif mode == "rev":
            d_inputs["x43_old"] += d_residuals["x43"]
            d_inputs["s43_i"] += delta_t * d_residuals["x43"]
            d_outputs["k43_i"] += (
                delta_t * butcher_diagonal_element * d_residuals["x43"]
            )
            d_outputs["x43"] -= d_residuals["x43"]
            exch_data = np.zeros(1)

            if self.comm.rank == 0:
                exch_data[0] = d_residuals["k43_i"][0]
                self.comm.Send(exch_data, dest=1, tag=2)

                d_outputs["k43_i"] -= d_residuals["k43_i"]
            elif self.comm.rank == 1:
                self.comm.Recv(exch_data, source=0, tag=2)
                d_outputs["x43"] += exch_data[0]
                d_inputs["x12"] += d_residuals["k43_i"]
                d_outputs["k43_i"] -= d_residuals["k43_i"]


def ode4d_analytical_solution(time, initial_values, initial_time=0.0):
    """Analytical solution to the above system of ODEs modelled by the two components"""
    a = np.sum(initial_values) / 4
    b = (np.sum(initial_values[0:3:2]) - np.sum(initial_values[1:4:2])) / 4
    c = (initial_values[3] - initial_values[1]) / 2
    d = (initial_values[0] - initial_values[2]) / 2
    passed_time = time - initial_time
    return np.array(
        [
            a * np.exp(passed_time)
            + b * np.exp(-passed_time)
            + c * np.sin(passed_time)
            + d * np.cos(passed_time),
            a * np.exp(passed_time)
            - b * np.exp(-passed_time)
            - c * np.cos(passed_time)
            + d * np.sin(passed_time),
            a * np.exp(passed_time)
            + b * np.exp(-passed_time)
            - c * np.sin(passed_time)
            - d * np.cos(passed_time),
            a * np.exp(passed_time)
            - b * np.exp(-passed_time)
            + c * np.cos(passed_time)
            - d * np.sin(passed_time),
        ]
    )


# The following components model the ODE system
#   d' = d
#   c' = c - d
#   b' = b + d
#   a' = a + b + c


class FirstParallelGroupChain(ExplicitUnsteadyComponent):
    """Models the first equation of the above system."""

    def setup(self):
        self.add_input("d_old", shape=1, tags=["d", "step_input_var"])
        self.add_input(
            "d_accumulated_stages", shape=1, tags=["d", "accumulated_stage_var"]
        )
        self.add_output("d_update", shape=1, tags=["d", "stage_output_var"])
        self.add_output("d_state", shape=1)

    def compute(self, inputs, outputs):
        delta_t = self.om_data_exchange.step_size
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        if butcher_diagonal_element == 0.0:
            factor = 1.0
        else:
            factor = 1 / (1 - delta_t * butcher_diagonal_element)
        old_influence = inputs["d_old"] + delta_t * inputs["d_accumulated_stages"]
        outputs["d_update"] = factor * old_influence
        outputs["d_state"] = factor * old_influence

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.om_data_exchange.step_size
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        if butcher_diagonal_element == 0.0:
            factor = 1.0
        else:
            factor = 1 / (1 - delta_t * butcher_diagonal_element)
        if mode == "fwd":
            if "d_update" in d_outputs:
                if "d_old" in d_inputs:
                    d_outputs["d_update"] += factor * d_inputs["d_old"]
                if "d_accumulated_stages" in d_inputs:
                    d_outputs["d_update"] += (
                        delta_t * factor * d_inputs["d_accumulated_stages"]
                    )
            if "d_state" in d_outputs:
                if "d_old" in d_inputs:
                    d_outputs["d_state"] += factor * d_inputs["d_old"]
                if "d_accumulated_stages" in d_inputs:
                    d_outputs["d_state"] += (
                        delta_t * factor * d_inputs["d_accumulated_stages"]
                    )
        elif mode == "rev":
            if "d_update" in d_outputs:
                if "d_old" in d_inputs:
                    d_inputs["d_old"] += factor * d_outputs["d_update"]
                if "d_accumulated_stages" in d_inputs:
                    d_inputs["d_accumulated_stages"] += (
                        delta_t * factor * d_outputs["d_update"]
                    )
            if "d_state" in d_outputs:
                if "d_old" in d_inputs:
                    d_inputs["d_old"] += factor * d_outputs["d_state"]
                if "d_accumulated_stages" in d_inputs:
                    d_inputs["d_accumulated_stages"] += (
                        delta_t * factor * d_outputs["d_state"]
                    )


class SecondParallelGroupChain1(ExplicitUnsteadyComponent):
    """Models the second equation of the above system."""

    def setup(self):
        self.add_input("c_old", shape=1, tags=["c", "step_input_var"])
        self.add_input(
            "c_accumulated_stages", shape=1, tags=["c", "accumulated_stage_var"]
        )
        self.add_input("d", shape=1)
        self.add_output("c_update", shape=1, tags=["c", "stage_output_var"])
        self.add_output("c_state", shape=1)

    def compute(self, inputs, outputs):
        delta_t = self.om_data_exchange.step_size
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        old_influence = inputs["c_old"] + delta_t * inputs["c_accumulated_stages"]
        if butcher_diagonal_element != 0.0:
            outputs["c_update"] = (old_influence - inputs["d"]) / (
                1 - butcher_diagonal_element * delta_t
            )
            outputs["c_state"] = (
                old_influence - delta_t * butcher_diagonal_element * inputs["d"]
            ) / (1 - butcher_diagonal_element * delta_t)
        else:
            outputs["c_update"] = old_influence - inputs["d"]
            outputs["c_state"] = old_influence

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.om_data_exchange.step_size
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        if butcher_diagonal_element != 0.0:
            factor = 1.0 / (1 - butcher_diagonal_element * delta_t)
        else:
            factor = 1.0
        if mode == "fwd":
            if "c_old" in d_inputs:
                if "c_state" in d_outputs:
                    d_outputs["c_state"] += factor * d_inputs["c_old"]
                if "c_update" in d_outputs:
                    d_outputs["c_update"] += factor * d_inputs["c_old"]
            if "c_accumulated_stages" in d_inputs:
                if "c_state" in d_outputs:
                    d_outputs["c_state"] += (
                        factor * delta_t * d_inputs["c_accumulated_stages"]
                    )
                if "c_update" in d_outputs:
                    d_outputs["c_update"] += (
                        factor * delta_t * d_inputs["c_accumulated_stages"]
                    )
            if "d" in d_inputs:
                if "c_update" in d_outputs:
                    d_outputs["c_update"] -= factor * d_inputs["d"]
                if "c_state" in d_outputs:
                    if butcher_diagonal_element != 0.0:
                        d_outputs["c_state"] -= (
                            factor * delta_t * butcher_diagonal_element * d_inputs["d"]
                        )
        if mode == "rev":
            if "c_old" in d_inputs:
                if "c_state" in d_outputs:
                    d_inputs["c_old"] += factor * d_outputs["c_state"]
                if "c_update" in d_outputs:
                    d_inputs["c_old"] += factor * d_outputs["c_update"]
            if "c_accumulated_stages" in d_inputs:
                if "c_state" in d_outputs:
                    d_inputs["c_accumulated_stages"] += (
                        factor * delta_t * d_outputs["c_state"]
                    )
                if "c_update" in d_outputs:
                    d_inputs["c_accumulated_stages"] += (
                        factor * delta_t * d_outputs["c_update"]
                    )
            if "d" in d_inputs:
                if "c_update" in d_outputs:
                    d_inputs["d"] -= factor * d_outputs["c_update"]
                if "c_state" in d_outputs:
                    if butcher_diagonal_element != 0.0:
                        d_inputs["d"] -= (
                            factor
                            * delta_t
                            * butcher_diagonal_element
                            * d_outputs["c_state"]
                        )


class SecondParallelGroupChain2(ExplicitUnsteadyComponent):
    """Models the third equation of the above system."""

    def setup(self):
        self.add_input("b_old", shape=1, tags=["b", "step_input_var"])
        self.add_input(
            "b_accumulated_stages", shape=1, tags=["b", "accumulated_stage_var"]
        )
        self.add_input("d", shape=1)
        self.add_output("b_update", shape=1, tags=["b", "stage_output_var"])
        self.add_output("b_state", shape=1)

    def compute(self, inputs, outputs):
        delta_t = self.om_data_exchange.step_size
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        old_influence = inputs["b_old"] + delta_t * inputs["b_accumulated_stages"]
        if butcher_diagonal_element != 0.0:
            outputs["b_update"] = (old_influence + inputs["d"]) / (
                1 - butcher_diagonal_element * delta_t
            )
            outputs["b_state"] = (
                old_influence + delta_t * butcher_diagonal_element * inputs["d"]
            ) / (1 - butcher_diagonal_element * delta_t)
        else:
            outputs["b_update"] = old_influence + inputs["d"]
            outputs["b_state"] = old_influence

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.om_data_exchange.step_size
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        if butcher_diagonal_element != 0.0:
            factor = 1.0 / (1 - butcher_diagonal_element * delta_t)
        else:
            factor = 1.0
        if mode == "fwd":
            if "b_old" in d_inputs:
                if "b_state" in d_outputs:
                    d_outputs["b_state"] += factor * d_inputs["b_old"]
                if "b_update" in d_outputs:
                    d_outputs["b_update"] += factor * d_inputs["b_old"]
            if "b_accumulated_stages" in d_inputs:
                if "b_state" in d_outputs:
                    d_outputs["b_state"] += (
                        factor * delta_t * d_inputs["b_accumulated_stages"]
                    )
                if "b_update" in d_outputs:
                    d_outputs["b_update"] += (
                        factor * delta_t * d_inputs["b_accumulated_stages"]
                    )
            if "d" in d_inputs:
                if "b_update" in d_outputs:
                    d_outputs["b_update"] += factor * d_inputs["d"]
                if "b_state" in d_outputs:
                    if butcher_diagonal_element != 0.0:
                        d_outputs["b_state"] += (
                            factor * delta_t * butcher_diagonal_element * d_inputs["d"]
                        )
        if mode == "rev":
            if "b_old" in d_inputs:
                if "b_state" in d_outputs:
                    d_inputs["b_old"] += factor * d_outputs["b_state"]
                if "b_update" in d_outputs:
                    d_inputs["b_old"] += factor * d_outputs["b_update"]
            if "b_accumulated_stages" in d_inputs:
                if "b_state" in d_outputs:
                    d_inputs["b_accumulated_stages"] += (
                        factor * delta_t * d_outputs["b_state"]
                    )
                if "b_update" in d_outputs:
                    d_inputs["b_accumulated_stages"] += (
                        factor * delta_t * d_outputs["b_update"]
                    )
            if "d" in d_inputs:
                if "b_update" in d_outputs:
                    d_inputs["d"] += factor * d_outputs["b_update"]
                if "b_state" in d_outputs:
                    if butcher_diagonal_element != 0.0:
                        d_inputs["d"] += (
                            factor
                            * delta_t
                            * butcher_diagonal_element
                            * d_outputs["b_state"]
                        )


class ThirdParallelGroupChain(ExplicitUnsteadyComponent):
    """Models the fourth equation of the above system."""

    def setup(self):
        self.add_input("a_old", shape=1, tags=["a", "step_input_var"])
        self.add_input(
            "a_accumulated_stages", shape=1, tags=["a", "accumulated_stage_var"]
        )
        self.add_input("b", shape=1)
        self.add_input("c", shape=1)
        self.add_output("a_update", shape=1, tags=["a", "stage_output_var"])
        self.add_output("a_state", shape=1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        delta_t = self.om_data_exchange.step_size
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        if butcher_diagonal_element != 0.0:
            factor = 1 / (1 - butcher_diagonal_element * delta_t)
        else:
            factor = 1.0
        old_influcence = inputs["a_old"] + delta_t * inputs["a_accumulated_stages"]
        outputs["a_update"] = factor * (old_influcence + inputs["b"] + inputs["c"])
        outputs["a_state"] = factor * (
            old_influcence
            + delta_t * butcher_diagonal_element * (inputs["b"] + inputs["c"])
        )

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None
    ):
        delta_t = self.om_data_exchange.step_size
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        if butcher_diagonal_element != 0.0:
            factor = 1 / (1 - butcher_diagonal_element * delta_t)
        else:
            factor = 1.0
        if mode == "fwd":
            if "a_old" in d_inputs:
                if "a_update" in d_outputs:
                    d_outputs["a_update"] += factor * d_inputs["a_old"]
                if "a_state" in d_outputs:
                    d_outputs["a_state"] += factor * d_inputs["a_old"]
            if "a_accumulated_stages" in d_inputs:
                if "a_update" in d_outputs:
                    d_outputs["a_update"] += (
                        factor * delta_t * d_inputs["a_accumulated_stages"]
                    )
                if "a_state" in d_outputs:
                    d_outputs["a_state"] += (
                        factor * delta_t * d_inputs["a_accumulated_stages"]
                    )
            if "b" in d_inputs:
                if "a_update" in d_outputs:
                    d_outputs["a_update"] += factor * d_inputs["b"]
                if "a_state" in d_outputs and butcher_diagonal_element != 0.0:
                    d_outputs["a_state"] += (
                        factor * butcher_diagonal_element * delta_t * d_inputs["b"]
                    )
            if "c" in d_inputs:
                if "a_update" in d_outputs:
                    d_outputs["a_update"] += factor * d_inputs["c"]
                if "a_state" in d_outputs and butcher_diagonal_element != 0.0:
                    d_outputs["a_state"] += (
                        factor * butcher_diagonal_element * delta_t * d_inputs["c"]
                    )
        if mode == "rev":
            if "a_old" in d_inputs:
                if "a_update" in d_outputs:
                    d_inputs["a_old"] += factor * d_outputs["a_update"]
                if "a_state" in d_outputs:
                    d_inputs["a_old"] += factor * d_outputs["a_state"]
            if "a_accumulated_stages" in d_inputs:
                if "a_update" in d_outputs:
                    d_inputs["a_accumulated_stages"] += (
                        factor * delta_t * d_outputs["a_update"]
                    )
                if "a_state" in d_outputs:
                    d_inputs["a_accumulated_stages"] += (
                        factor * delta_t * d_outputs["a_state"]
                    )
            if "b" in d_inputs:
                if "a_update" in d_outputs:
                    d_inputs["b"] += factor * d_outputs["a_update"]
                if "a_state" in d_outputs and butcher_diagonal_element != 0.0:
                    d_inputs["b"] += (
                        factor
                        * butcher_diagonal_element
                        * delta_t
                        * d_outputs["a_state"]
                    )
            if "c" in d_inputs:
                if "a_update" in d_outputs:
                    d_inputs["c"] += factor * d_outputs["a_update"]
                if "a_state" in d_outputs and butcher_diagonal_element != 0.0:
                    d_inputs["c"] += (
                        factor
                        * butcher_diagonal_element
                        * delta_t
                        * d_outputs["a_state"]
                    )


def parallel_group_chain_solution(time: float, initial_values: np.ndarray):
    """
    Analytical solution to the above ODE system. Expects in order d to a,
    and returns in the same order
    """
    return np.array(
        [
            initial_values[0] * np.exp(time),
            (initial_values[1] - time * initial_values[0]) * np.exp(time),
            (initial_values[2] + time * initial_values[0]) * np.exp(time),
            (initial_values[3] + time * (initial_values[1] + initial_values[2]))
            * np.exp(time),
        ]
    )
