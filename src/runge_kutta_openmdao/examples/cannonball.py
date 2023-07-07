import numpy as np
import openmdao.api as om

from scipy.constants import g as gravity_accel

from runge_kutta_openmdao.runge_kutta.butcher_tableau import ButcherTableau
from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl
from runge_kutta_openmdao.runge_kutta.runge_kutta_integrator import RungeKuttaIntegrator


class VelocityComponent(om.ExplicitComponent):
    def setup(self):
        self.add_input("aircraft_mass", val=1.0)
        self.add_input("thrust", val=1.0)
        self.add_input("angle_of_attack", val=0.0)
        self.add_input("drag", val=0.0)
        self.add_input("flight_path_angle", val=0.0)

        self.add_output("velocity_slope", tags=["stage_output_var", "velocity"])

    def compute(self, inputs, outputs):
        outputs["velocity_slope"] = (
            inputs["thrust"] * np.cos(inputs["angle_of_attack"]) - inputs["drag"]
        ) / inputs["aircraft_mass"] - gravity_accel * np.sin(
            inputs["flight_path_angle"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            d_outputs["velocity_slope"] -= (
                (inputs["thrust"] * np.cos(inputs["angle_of_attack"]) - inputs["drag"])
                / inputs["aircraft_mass"] ** 2
                * d_inputs["aircraft_mass"]
            )
            d_outputs["velocity_slope"] += (
                d_inputs["thrust"]
                * np.cos(inputs["angle_of_attack"])
                / inputs["aircraft_mass"]
            )
            d_outputs["velocity_slope"] -= (
                inputs["thrust"]
                * np.sin(inputs["angle_of_attack"])
                * d_inputs["angle_of_attack"]
                / inputs["aircraft_mass"]
            )
            d_outputs["velocity_slope"] -= d_inputs["drag"] / inputs["aircraft_mass"]
            d_outputs["velocity_slope"] -= (
                gravity_accel
                * np.cos(inputs["flight_path_angle"])
                * d_inputs["flight_path_angle"]
            )
        elif mode == "rev":
            d_inputs["aircraft_mass"] -= (
                (inputs["thrust"] * np.cos(inputs["angle_of_attack"]) - inputs["drag"])
                / inputs["aircraft_mass"] ** 2
                * d_outputs["velocity_slope"]
            )
            d_inputs["thrust"] += (
                d_outputs["velocity_slope"]
                * np.cos(inputs["angle_of_attack"])
                / inputs["aircraft_mass"]
            )
            d_inputs["angle_of_attack"] -= (
                inputs["thrust"]
                * np.sin(inputs["angle_of_attack"])
                * d_outputs["velocity_slope"]
                / inputs["aircraft_mass"]
            )
            d_inputs["drag"] -= d_outputs["velocity_slope"] / inputs["aircraft_mass"]
            d_inputs["flight_path_angle"] -= (
                gravity_accel
                * np.cos(inputs["flight_path_angle"])
                * d_outputs["velocity_slope"]
            )


class VelocityAssembler(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("velocity_old", val=1.0, tags=["step_input_var", "velocity"])
        self.add_input(
            "velocity_accumulated_stages",
            val=0.0,
            tags=["accumulated_stage_var", "velocity"],
        )
        self.add_input("velocity_slope", val=0.0)

        self.add_output("velocity", val=1.0)

    def compute(self, inputs, outputs):
        outputs["velocity"] = inputs["velocity_old"] + self.options[
            "integration_control"
        ].delta_t * (
            inputs["velocity_accumulated_stages"]
            + self.options["integration_control"].butcher_diagonal_element
            * inputs["velocity_slope"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if mode == "fwd":
            d_outputs["velocity"] += d_inputs["velocity_old"]
            d_outputs["velocity"] += delta_t * d_inputs["velocity_accumulated_stages"]
            d_outputs["velocity"] += (
                delta_t * butcher_diagonal_element * d_inputs["velocity_slope"]
            )
        elif mode == "rev":
            d_inputs["velocity_old"] += d_outputs["velocity"]
            d_inputs["velocity_accumulated_stages"] += delta_t * d_outputs["velocity"]
            d_inputs["velocity_slope"] += (
                delta_t * butcher_diagonal_element * d_outputs["velocity"]
            )


class FlightPathAngleComponent(om.ExplicitComponent):
    def setup(self):
        self.add_input("aircraft_mass", val=1.0)
        self.add_input("velocity", val=1.0)
        self.add_input("thrust", val=1.0)
        self.add_input("angle_of_attack", val=0.0)
        self.add_input("lift", val=0.0)
        self.add_input("flight_path_angle")

        self.add_output(
            "flight_path_angle_slope", tags=["stage_output_var", "flight_path_angle"]
        )

    def compute(self, inputs, outputs):
        outputs["flight_path_angle_slope"] = (
            (
                (inputs["thrust"] * np.sin(inputs["angle_of_attack"]) + inputs["lift"])
                / inputs["aircraft_mass"]
            )
            - gravity_accel * np.cos(inputs["flight_path_angle"])
        ) / inputs["velocity"]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            d_outputs["flight_path_angle_slope"] -= d_inputs["aircraft_mass"] * (
                (inputs["thrust"] * np.sin(inputs["angle_of_attack"]) + inputs["lift"])
                / (inputs["aircraft_mass"] ** 2 * inputs["velocity"])
            )
            d_outputs["flight_path_angle_slope"] -= (
                d_inputs["velocity"]
                * (
                    (
                        (
                            inputs["thrust"] * np.sin(inputs["angle_of_attack"])
                            + inputs["lift"]
                        )
                        / inputs["aircraft_mass"]
                    )
                    - gravity_accel * np.cos(inputs["flight_path_angle"])
                )
                / inputs["velocity"] ** 2
            )
            d_outputs["flight_path_angle_slope"] += (
                d_inputs["thrust"] * np.sin(inputs["angle_of_attack"])
            ) / (inputs["aircraft_mass"] * inputs["velocity"])

            d_outputs["flight_path_angle_slope"] += (
                inputs["thrust"]
                * np.cos(inputs["angle_of_attack"])
                * d_inputs["angle_of_attack"]
            ) / (inputs["aircraft_mass"] * inputs["velocity"])

            d_outputs["flight_path_angle_slope"] += d_inputs["lift"] / (
                inputs["aircraft_mass"] * inputs["velocity"]
            )

            d_outputs["flight_path_angle_slope"] += (
                gravity_accel
                * np.sin(inputs["flight_path_angle"])
                * d_inputs["flight_path_angle"]
            ) / inputs["velocity"]
        elif mode == "rev":
            d_inputs["aircraft_mass"] -= d_outputs["flight_path_angle_slope"] * (
                (inputs["thrust"] * np.sin(inputs["angle_of_attack"]) + inputs["lift"])
                / (inputs["aircraft_mass"] ** 2 * inputs["velocity"])
            )
            d_inputs["velocity"] -= (
                d_outputs["flight_path_angle_slope"]
                * (
                    (
                        (
                            inputs["thrust"] * np.sin(inputs["angle_of_attack"])
                            + inputs["lift"]
                        )
                        / inputs["aircraft_mass"]
                    )
                    - gravity_accel * np.cos(inputs["flight_path_angle"])
                )
                / inputs["velocity"] ** 2
            )
            d_inputs["thrust"] += (
                d_outputs["flight_path_angle_slope"] * np.sin(inputs["angle_of_attack"])
            ) / (inputs["aircraft_mass"] * inputs["velocity"])

            d_inputs["angle_of_attack"] += (
                inputs["thrust"]
                * np.cos(inputs["angle_of_attack"])
                * d_outputs["flight_path_angle_slope"]
            ) / (inputs["aircraft_mass"] * inputs["velocity"])

            d_inputs["lift"] += d_outputs["flight_path_angle_slope"] / (
                inputs["aircraft_mass"] * inputs["velocity"]
            )

            d_inputs["flight_path_angle"] += (
                gravity_accel
                * np.sin(inputs["flight_path_angle"])
                * d_outputs["flight_path_angle_slope"]
            ) / inputs["velocity"]


class FlightPathAngleAssembler(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input(
            "flight_path_angle_old",
            val=0.0,
            tags=["step_input_var", "flight_path_angle"],
        )
        self.add_input(
            "flight_path_angle_accumulated_stages",
            val=0.0,
            tags=["accumulated_stage_var", "flight_path_angle"],
        )
        self.add_input("flight_path_angle_slope", val=0.0)

        self.add_output("flight_path_angle", val=1.0)

    def compute(self, inputs, outputs):
        outputs["flight_path_angle"] = inputs["flight_path_angle_old"] + self.options[
            "integration_control"
        ].delta_t * (
            inputs["flight_path_angle_accumulated_stages"]
            + self.options["integration_control"].butcher_diagonal_element
            * inputs["flight_path_angle_slope"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if mode == "fwd":
            d_outputs["flight_path_angle"] += d_inputs["flight_path_angle_old"]
            d_outputs["flight_path_angle"] += (
                delta_t * d_inputs["flight_path_angle_accumulated_stages"]
            )
            d_outputs["flight_path_angle"] += (
                delta_t * butcher_diagonal_element * d_inputs["flight_path_angle_slope"]
            )
        elif mode == "rev":
            d_inputs["flight_path_angle_old"] += d_outputs["flight_path_angle"]
            d_inputs["flight_path_angle_accumulated_stages"] += (
                delta_t * d_outputs["flight_path_angle"]
            )
            d_inputs["flight_path_angle_slope"] += (
                delta_t * butcher_diagonal_element * d_outputs["flight_path_angle"]
            )


class AltitudeComponent(om.ExplicitComponent):
    def setup(self):
        self.add_input("velocity", val=1.0)
        self.add_input("flight_path_angle", val=0.0)
        # The next two do nothing, but are required for the RK-Integrator
        # self.add_input("altitude_old", tags=["step_input_var", "altitude"])
        # self.add_input(
        #     "altitude_accumulated_stages", tags=["accumulated_stage_var", "altitude"]
        # )

        self.add_output(
            "altitude_slope", val=0.0, tags=["stage_output_var", "altitude"]
        )

    def compute(self, inputs, outputs):
        outputs["altitude_slope"] = inputs["velocity"] * np.sin(
            inputs["flight_path_angle"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            d_outputs["altitude_slope"] += d_inputs["velocity"] * np.sin(
                inputs["flight_path_angle"]
            )
            d_outputs["altitude_slope"] += (
                inputs["velocity"]
                * np.cos(inputs["flight_path_angle"])
                * d_inputs["flight_path_angle"]
            )
        elif mode == "rev":
            d_inputs["velocity"] += d_outputs["altitude_slope"] * np.sin(
                inputs["flight_path_angle"]
            )
            d_inputs["flight_path_angle"] += (
                inputs["velocity"]
                * np.cos(inputs["flight_path_angle"])
                * d_outputs["altitude_slope"]
            )


class RangeComponent(om.ExplicitComponent):
    def setup(self):
        self.add_input("velocity", val=1.0)
        self.add_input("flight_path_angle", val=0.0)
        # The next two do nothing, but are required for the RK-Integrator
        # self.add_input("range_old", tags=["step_input_var", "range"])
        # self.add_input(
        #     "range_accumulated_stages", tags=["accumulated_stage_var", "range"]
        # )

        self.add_output("range_slope", val=0.0, tags=["stage_output_var", "range"])

    def compute(self, inputs, outputs):
        outputs["range_slope"] = inputs["velocity"] * np.cos(
            inputs["flight_path_angle"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            d_outputs["range_slope"] += d_inputs["velocity"] * np.cos(
                inputs["flight_path_angle"]
            )
            d_outputs["range_slope"] -= (
                inputs["velocity"]
                * np.sin(inputs["flight_path_angle"])
                * d_inputs["flight_path_angle"]
            )
        elif mode == "rev":
            d_inputs["velocity"] += d_outputs["range_slope"] * np.cos(
                inputs["flight_path_angle"]
            )
            d_inputs["flight_path_angle"] -= (
                inputs["velocity"]
                * np.sin(inputs["flight_path_angle"])
                * d_outputs["range_slope"]
            )


# The following two classes are copied from the corresponding example in the OpenMDAO documentation
# https://openmdao.org/newdocs/versions/latest/advanced_user_guide/example/euler_integration_example.html
# They are augmented with a compute_jacvec_product_function


class DynamicPressureComp(om.ExplicitComponent):
    def setup(self):
        self.add_input(name="rho", val=1.0, units="kg/m**3", desc="atmospheric density")
        self.add_input(name="v", val=1.0, units="m/s", desc="air-relative velocity")

        self.add_output(name="q", val=1.0, units="N/m**2", desc="dynamic pressure")

    def compute(self, inputs, outputs):
        outputs["q"] = 0.5 * inputs["rho"] * inputs["v"] ** 2

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            d_outputs["q"] += 0.5 * d_inputs["rho"] * inputs["v"] ** 2
            d_outputs["q"] += inputs["rho"] * inputs["v"] * d_inputs["v"]
        elif mode == "rev":
            d_inputs["rho"] += 0.5 * d_outputs["q"] * inputs["v"] ** 2
            d_inputs["v"] += inputs["rho"] * inputs["v"] * d_outputs["q"]


class LiftDragForceComp(om.ExplicitComponent):
    """
    Compute the aerodynamic forces on the vehicle in the wind axis frame
    (lift, drag, cross) force.
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        self.add_input(name="CL", val=0.0, desc="lift coefficient")
        self.add_input(name="CD", val=0.0, desc="drag coefficient")
        self.add_input(name="q", val=0.0, units="N/m**2", desc="dynamic pressure")
        self.add_input(
            name="S", val=0.0, units="m**2", desc="aerodynamic reference area"
        )

        self.add_output(
            name="f_lift", shape=(1,), units="N", desc="aerodynamic lift force"
        )
        self.add_output(
            name="f_drag", shape=(1,), units="N", desc="aerodynamic drag force"
        )

    def compute(self, inputs, outputs):
        q = inputs["q"]
        S = inputs["S"]
        CL = inputs["CL"]
        CD = inputs["CD"]

        qS = q * S

        outputs["f_lift"] = qS * CL
        outputs["f_drag"] = qS * CD

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            d_outputs["f_lift"] += d_inputs["q"] * inputs["S"] * inputs["CL"]
            d_outputs["f_lift"] += inputs["q"] * d_inputs["S"] * inputs["CL"]
            d_outputs["f_lift"] += inputs["q"] * inputs["S"] * d_inputs["CL"]

            d_outputs["f_drag"] += d_inputs["q"] * inputs["S"] * inputs["CD"]
            d_outputs["f_drag"] += inputs["q"] * d_inputs["S"] * inputs["CD"]
            d_outputs["f_drag"] += inputs["q"] * inputs["S"] * d_inputs["CD"]
        elif mode == "rev":
            d_inputs["q"] += d_outputs["f_lift"] * inputs["S"] * inputs["CL"]
            d_inputs["S"] += inputs["q"] * d_outputs["f_lift"] * inputs["CL"]
            d_inputs["CL"] += inputs["q"] * inputs["S"] * d_outputs["f_lift"]

            d_inputs["q"] += d_outputs["f_drag"] * inputs["S"] * inputs["CD"]
            d_inputs["S"] += inputs["q"] * d_outputs["f_drag"] * inputs["CD"]
            d_inputs["CD"] += inputs["q"] * inputs["S"] * d_outputs["f_drag"]


if __name__ == "__main__":
    integration_control = IntegrationControl(0.0, 100, 1e-1)
    cannon_prob = om.Problem()

    velocity_angle_group = om.Group()

    pressure_velocity_group = om.Group()
    pressure_velocity_group.add_subsystem("DynamicPressure", DynamicPressureComp())
    pressure_velocity_group.add_subsystem("LiftDragForce", LiftDragForceComp())
    pressure_velocity_group.add_subsystem(
        "velocity", VelocityComponent(), promotes=["aircraft_mass"]
    )
    pressure_velocity_group.add_subsystem(
        "velocity_assembler", VelocityAssembler(integration_control=integration_control)
    )
    pressure_velocity_group.NonlinearSolver = om.NewtonSolver(solve_subsystems=False)

    velocity_angle_group.add_subsystem(
        "pressure_velocity", pressure_velocity_group, promotes=["*"]
    )

    flight_angle_group = om.Group()
    flight_angle_group.add_subsystem(
        "flight_path_angle", FlightPathAngleComponent(), promotes=["aircraft_mass"]
    )
    flight_angle_group.add_subsystem(
        "flight_path_angle_assembler",
        FlightPathAngleAssembler(integration_control=integration_control),
    )
    flight_angle_group.NonlinearSolver = om.NewtonSolver(solve_subsystems=False)

    velocity_angle_group.add_subsystem(
        "flight_angle", flight_angle_group, promotes=["*"]
    )
    velocity_angle_group.NonlinearSolver = om.NewtonSolver(solve_subsystems=True)

    cannon_prob.model.add_subsystem(
        "velocity_angle", velocity_angle_group, promotes=["*"]
    )

    cannon_prob.model.add_subsystem("altitude", AltitudeComponent())
    cannon_prob.model.add_subsystem("range", RangeComponent())

    cannon_prob.model.connect("DynamicPressure.q", "LiftDragForce.q")

    cannon_prob.model.connect("LiftDragForce.f_lift", "flight_path_angle.lift")
    cannon_prob.model.connect("LiftDragForce.f_drag", "velocity.drag")

    cannon_prob.model.connect(
        "velocity.velocity_slope", "velocity_assembler.velocity_slope"
    )

    cannon_prob.model.connect("velocity_assembler.velocity", "DynamicPressure.v")
    cannon_prob.model.connect(
        "velocity_assembler.velocity", "flight_path_angle.velocity"
    )
    cannon_prob.model.connect("velocity_assembler.velocity", "altitude.velocity")
    cannon_prob.model.connect("velocity_assembler.velocity", "range.velocity")

    cannon_prob.model.connect(
        "flight_path_angle.flight_path_angle_slope",
        "flight_path_angle_assembler.flight_path_angle_slope",
    )

    cannon_prob.model.connect(
        "flight_path_angle_assembler.flight_path_angle", "velocity.flight_path_angle"
    )
    cannon_prob.model.connect(
        "flight_path_angle_assembler.flight_path_angle",
        "flight_path_angle.flight_path_angle",
    )

    cannon_prob.model.connect(
        "flight_path_angle_assembler.flight_path_angle", "altitude.flight_path_angle"
    )
    cannon_prob.model.connect(
        "flight_path_angle_assembler.flight_path_angle", "range.flight_path_angle"
    )

    cannon_prob.model.nonlinear_solver = om.NonlinearRunOnce()
    cannon_prob.model.linear_solver = om.PETScKrylov()
    # cannon_prob.model.linear_solver.precon = om.LinearRunOnce()

    cannon_prob.setup()

    cannon_prob.set_val("LiftDragForce.CL", 0.0)
    cannon_prob.set_val("LiftDragForce.CD", 0.05)
    cannon_prob.set_val("LiftDragForce.S", 0.25 * np.pi)
    cannon_prob.set_val("DynamicPressure.rho", 1.225)
    cannon_prob.set_val("aircraft_mass", 5.5)

    outer_prob = om.Problem()

    gamma = (2.0 - np.sqrt(2.0)) / 2.0
    butcher_tableau = ButcherTableau(
        np.array(
            [
                [gamma, 0.0],
                [1 - gamma, gamma],
            ]
        ),
        np.array([1 - gamma, gamma]),
        np.array([gamma, 1.0]),
    )

    outer_prob.model.add_subsystem(
        "RK",
        RungeKuttaIntegrator(
            time_stage_problem=cannon_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=[
                "velocity",
                "flight_path_angle",
                "altitude",
                "range",
            ],
            write_out_distance=10,
            write_file="cannonball.h5",
        ),
    )

    # outer_prob.driver = om.ScipyOptimizeDriver()
    # outer_prob.driver.options["optimizer"] = "SLSQP"
    # outer_prob.model.add_objective("RK.range_final", ref=-1.0)
    # outer_prob.model.add_design_var("RK.flight_path_angle_initial")
    #
    outer_prob.setup()
    outer_prob.run_model()

    cannon_prob.check_partials()
    #
    # outer_prob.run_driver()

    # TODO: investigate why finite differences are not good. Are they even applicable to RK methods
    # outer_prob.check_partials()
