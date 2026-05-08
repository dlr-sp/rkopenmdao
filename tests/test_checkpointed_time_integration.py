import pytest

from rkopenmdao.callback import Callback
from rkopenmdao.error_controller import ErrorControllerConfig
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.termination_criterion import (
    PredefinedNumberOfSteps,
    PredefinedFinalTime,
)
from rkopenmdao.checkpointed_time_integration.no_checkpoint_time_integration import (
    NoCheckpointTimeIntegration,
)
from rkopenmdao.checkpointed_time_integration.all_checkpoint_time_integration import (
    AllCheckpointTimeIntegration,
)
from rkopenmdao.checkpointed_time_integration.pyrevolve_time_integration import (
    PyrevolveTimeIntegration,
)

from rkopenmdao.error_controllers import pseudo, integral
from rkopenmdao.error_measurer import SimpleErrorMeasurer

from .utils.mock_classes import MockODE, MockDiscretization
from .utils.time_integration_test_utils import (
    AbstractTestHomogeneousTimeIntegrationSystem,
    AbstractTestTimeIntegrationUnit,
    AbstractTestAdaptiveTimeIntegrationSystem,
)


@pytest.fixture
def homogeneous_error_controller_and_measurer():
    return (pseudo(1), SimpleErrorMeasurer())


class TestNoCheckpointTimeIntegrationUnit(AbstractTestTimeIntegrationUnit):
    @pytest.fixture
    def time_integrator(self, homogeneous_error_controller_and_measurer):
        return NoCheckpointTimeIntegration(
            MockODE(),
            MockDiscretization(),
            homogeneous_error_controller_and_measurer[0],
            homogeneous_error_controller_and_measurer[1],
            IntegrationConfig(False, PredefinedNumberOfSteps(5), 1.0),
            [],
            [],
            [],
        )

    def test_integrate_adjoint_derivative(
        self, time_integrator, initial_state, final_state_perturbations
    ):
        with pytest.raises(NotImplementedError):
            super().test_integrate_adjoint_derivative(
                time_integrator, initial_state, final_state_perturbations
            )


class TestAllCheckpointTimeIntegrationUnit(AbstractTestTimeIntegrationUnit):
    @pytest.fixture
    def time_integrator(self, homogeneous_error_controller_and_measurer):
        return AllCheckpointTimeIntegration(
            MockODE(),
            MockDiscretization(),
            homogeneous_error_controller_and_measurer[0],
            homogeneous_error_controller_and_measurer[1],
            IntegrationConfig(False, PredefinedNumberOfSteps(5), 1.0),
            [],
            [],
            [],
        )


class TestPyrevolveTimeIntegrationUnit(AbstractTestTimeIntegrationUnit):
    @pytest.fixture
    def time_integrator(self, homogeneous_error_controller_and_measurer):
        return PyrevolveTimeIntegration(
            MockODE(),
            MockDiscretization(),
            homogeneous_error_controller_and_measurer[0],
            homogeneous_error_controller_and_measurer[1],
            IntegrationConfig(False, PredefinedNumberOfSteps(5), 1.0),
            [],
            [],
            [],
        )

    def test_setup_revolver_class_error(
        self, time_integrator: PyrevolveTimeIntegration
    ):
        with pytest.raises(TypeError):
            time_integrator._setup_revolver_class_type("foo")


@pytest.fixture
def homogeneous_time_integration_test_case(
    ode_with_reference_state_and_solution,
    discretization_order_pair,
    homogeneous_error_controller_and_measurer,
):
    return (
        ode_with_reference_state_and_solution,
        discretization_order_pair,
        homogeneous_error_controller_and_measurer,
    )


class AbstractTestHomogeneousCheckpointedTimeIntegrationSystem(
    AbstractTestHomogeneousTimeIntegrationSystem
):
    @pytest.fixture
    def initial_state(self, homogeneous_time_integration_test_case):
        return homogeneous_time_integration_test_case[0].initial_values

    @pytest.fixture
    def initial_state_perturbations(self, homogeneous_time_integration_test_case):
        return homogeneous_time_integration_test_case[0].initial_value_perturbations

    @pytest.fixture
    def final_state_perturbations(self, homogeneous_time_integration_test_case):
        return homogeneous_time_integration_test_case[0].final_value_perturbations

    @pytest.fixture
    def expected_order(self, homogeneous_time_integration_test_case):
        return min(
            homogeneous_time_integration_test_case[1].order,
            homogeneous_time_integration_test_case[0].order_barrier,
        )

    @pytest.fixture
    def reference_solution(self, homogeneous_time_integration_test_case):
        return homogeneous_time_integration_test_case[0].reference_solution(
            homogeneous_time_integration_test_case[0].initial_values, 1.0
        )

    @pytest.fixture
    def reference_derivative(self, homogeneous_time_integration_test_case):
        return homogeneous_time_integration_test_case[0].reference_derivative(
            homogeneous_time_integration_test_case[0].initial_values,
            homogeneous_time_integration_test_case[0].initial_value_perturbations,
            1.0,
        )

    @pytest.fixture
    def reference_adjoint_derivative(self, homogeneous_time_integration_test_case):
        return homogeneous_time_integration_test_case[0].reference_adjoint_derivative(
            homogeneous_time_integration_test_case[0].initial_values,
            homogeneous_time_integration_test_case[0].final_value_perturbations,
            1.0,
        )


class TestHomogeneousNoCheckpointTimeIntegrationSystem(
    AbstractTestHomogeneousCheckpointedTimeIntegrationSystem
):

    @pytest.fixture
    def time_integrator_creator(self, homogeneous_time_integration_test_case):
        return lambda step_size: NoCheckpointTimeIntegration(
            ode=homogeneous_time_integration_test_case[0].ode,
            time_discretization_scheme=homogeneous_time_integration_test_case[
                1
            ].time_discretization,
            error_controller=homogeneous_time_integration_test_case[2][0],
            error_measurer=homogeneous_time_integration_test_case[2][1],
            time_integration_config=IntegrationConfig(
                False, PredefinedNumberOfSteps(int(1 / step_size)), step_size
            ),
            integrate_callbacks=[],
            integrate_derivative_callbacks=[],
            integrate_adjoint_derivative_callbacks=[],
        )

    def test_integrate_adjoint_derivative_order(
        self,
        time_integrator_creator,
        initial_state,
        final_state_perturbations,
        expected_order,
        reference_adjoint_derivative,
    ):
        with pytest.raises(NotImplementedError):
            return super().test_integrate_adjoint_derivative_order(
                time_integrator_creator,
                initial_state,
                final_state_perturbations,
                expected_order,
                reference_adjoint_derivative,
            )

    def test_derivative_duality(
        self,
        time_integrator_creator,
        initial_state,
        initial_state_perturbations,
        final_state_perturbations,
    ):
        with pytest.raises(NotImplementedError):
            super().test_derivative_duality(
                time_integrator_creator,
                initial_state,
                initial_state_perturbations,
                final_state_perturbations,
            )


class TestHomogeneousAllCheckpointTimeIntegrationSystem(
    AbstractTestHomogeneousCheckpointedTimeIntegrationSystem
):
    @pytest.fixture
    def time_integrator_creator(self, homogeneous_time_integration_test_case):
        return lambda step_size: AllCheckpointTimeIntegration(
            ode=homogeneous_time_integration_test_case[0].ode,
            time_discretization_scheme=homogeneous_time_integration_test_case[
                1
            ].time_discretization,
            error_controller=homogeneous_time_integration_test_case[2][0],
            error_measurer=homogeneous_time_integration_test_case[2][1],
            time_integration_config=IntegrationConfig(
                False, PredefinedNumberOfSteps(int(1 / step_size)), step_size
            ),
            integrate_callbacks=[],
            integrate_derivative_callbacks=[],
            integrate_adjoint_derivative_callbacks=[],
        )


class TestHomogeneousPyrevolveTimeIntegrationSystem(
    AbstractTestHomogeneousCheckpointedTimeIntegrationSystem
):
    @pytest.fixture
    def time_integrator_creator(self, homogeneous_time_integration_test_case):
        return lambda step_size: PyrevolveTimeIntegration(
            ode=homogeneous_time_integration_test_case[0].ode,
            time_discretization_scheme=homogeneous_time_integration_test_case[
                1
            ].time_discretization,
            error_controller=homogeneous_time_integration_test_case[2][0],
            error_measurer=homogeneous_time_integration_test_case[2][1],
            time_integration_config=IntegrationConfig(
                False, PredefinedNumberOfSteps(int(1 / step_size)), step_size
            ),
            integrate_callbacks=[],
            integrate_derivative_callbacks=[],
            integrate_adjoint_derivative_callbacks=[],
        )


# @pytest.fixture
# def adaptive_error_controller_and_measurer():
#     return (
#         lambda p: integral(p, config=ErrorControllerConfig(tol=1e-3)),
#         SimpleErrorMeasurer(),
#     )


@pytest.fixture
def adaptive_time_integration_test_case(
    ode_with_reference_state_and_solution_for_adaptive,
    adaptive_discretization_order_pair,
    adaptive_error_controller_and_measurer,
):
    return (
        ode_with_reference_state_and_solution_for_adaptive,
        adaptive_discretization_order_pair,
        adaptive_error_controller_and_measurer,
    )


class AbstractTestAdaptiveCheckpointedTimeIntegrationSystem(
    AbstractTestAdaptiveTimeIntegrationSystem
):
    @pytest.fixture
    def initial_state(self, adaptive_time_integration_test_case):
        return adaptive_time_integration_test_case[0].initial_values

    @pytest.fixture
    def initial_state_perturbations(self, adaptive_time_integration_test_case):
        return adaptive_time_integration_test_case[0].initial_value_perturbations

    @pytest.fixture
    def final_state_perturbations(self, adaptive_time_integration_test_case):
        return adaptive_time_integration_test_case[0].final_value_perturbations

    @pytest.fixture
    def expected_order(self, adaptive_time_integration_test_case):
        return min(
            adaptive_time_integration_test_case[1].order,
            adaptive_time_integration_test_case[0].order_barrier,
        )

    @pytest.fixture
    def reference_solution(self, adaptive_time_integration_test_case):
        return adaptive_time_integration_test_case[0].reference_solution(
            adaptive_time_integration_test_case[0].initial_values, 1.0
        )

    @pytest.fixture
    def reference_derivative(self, adaptive_time_integration_test_case):
        return adaptive_time_integration_test_case[0].reference_derivative(
            adaptive_time_integration_test_case[0].initial_values,
            adaptive_time_integration_test_case[0].initial_value_perturbations,
            1.0,
        )

    @pytest.fixture
    def reference_adjoint_derivative(self, adaptive_time_integration_test_case):
        return adaptive_time_integration_test_case[0].reference_adjoint_derivative(
            adaptive_time_integration_test_case[0].initial_values,
            adaptive_time_integration_test_case[0].final_value_perturbations,
            1.0,
        )


class RecordStepSizes(Callback):
    def __init__(self):
        self.step_sizes = set()

    def after_iteration(
        self, iteration, time_integration_state, ode, discretization_scheme
    ):
        self.step_sizes.add(time_integration_state.step_size_history[0])


class TestAdaptiveNoCheckpointTimeIntegrationSystem(
    AbstractTestAdaptiveCheckpointedTimeIntegrationSystem
):

    @pytest.fixture
    def time_integrator_creator(self, adaptive_time_integration_test_case):
        return lambda step_size: NoCheckpointTimeIntegration(
            ode=adaptive_time_integration_test_case[0].ode,
            time_discretization_scheme=adaptive_time_integration_test_case[
                1
            ].time_discretization,
            error_controller=adaptive_time_integration_test_case[2][0](
                adaptive_time_integration_test_case[1].order
            ),
            error_measurer=adaptive_time_integration_test_case[2][1],
            time_integration_config=IntegrationConfig(
                True,
                PredefinedFinalTime(
                    1.0
                    + adaptive_time_integration_test_case[0].initial_values.initial_time
                ),
                step_size,
            ),
            integrate_callbacks=[RecordStepSizes()],
            integrate_derivative_callbacks=[],
            integrate_adjoint_derivative_callbacks=[],
        )

    def test_derivative_duality(
        self,
        time_integrator_creator,
        initial_state,
        initial_state_perturbations,
        final_state_perturbations,
    ):
        with pytest.raises(NotImplementedError):
            super().test_derivative_duality(
                time_integrator_creator,
                initial_state,
                initial_state_perturbations,
                final_state_perturbations,
            )


class TestAdaptiveAllCheckpointTimeIntegrationSystem(
    AbstractTestAdaptiveCheckpointedTimeIntegrationSystem
):

    @pytest.fixture
    def time_integrator_creator(self, adaptive_time_integration_test_case):
        return lambda step_size: AllCheckpointTimeIntegration(
            ode=adaptive_time_integration_test_case[0].ode,
            time_discretization_scheme=adaptive_time_integration_test_case[
                1
            ].time_discretization,
            error_controller=adaptive_time_integration_test_case[2][0](
                adaptive_time_integration_test_case[1].order
            ),
            error_measurer=adaptive_time_integration_test_case[2][1],
            time_integration_config=IntegrationConfig(
                True,
                PredefinedFinalTime(
                    1.0
                    + adaptive_time_integration_test_case[0].initial_values.initial_time
                ),
                step_size,
            ),
            integrate_callbacks=[RecordStepSizes()],
            integrate_derivative_callbacks=[],
            integrate_adjoint_derivative_callbacks=[],
        )
