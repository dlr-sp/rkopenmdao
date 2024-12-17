
import openmdao.api as om
import h5py
import pytest
import numpy as np

from rkopenmdao.integration_control import IntegrationControl, TerminationCriterion
from rkopenmdao.butcher_tableaux import (
    embedded_third_order_four_stage_esdirk,
)
from rkopenmdao.file_writer import TXTFileWriter
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from .test_components import TestComp1
from .test_postprocessing_problems import SquaringComponent

write_out_distance = 1
write_file = "a.txt"
other_file = "b.txt"
test_prob = om.Problem()
termination_criterion = TerminationCriterion('num_steps', 100)
integration_control = IntegrationControl(1.0, termination_criterion, 0.01)

butcher_tableau = embedded_third_order_four_stage_esdirk

test_prob.model.add_subsystem(
    "test_comp", TestComp1(integration_control=integration_control)
)

time_int_prob = om.Problem()
time_int_prob.model.add_subsystem(
    "rk_integration",
    RungeKuttaIntegrator(
        time_stage_problem=test_prob,
        butcher_tableau=butcher_tableau,
        integration_control=integration_control,
        write_out_distance=write_out_distance,
        write_file=write_file,
        file_writing_implementation=TXTFileWriter,
        time_integration_quantities=["x"],
    ),
)

time_int_prob.setup()
time_int_prob.run_model()