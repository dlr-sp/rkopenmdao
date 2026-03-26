"""
This file containsOpenMDAO component implementations that facilitate the use of time
integration with RKOpenMDAO.
"""

from dataclasses import dataclass, field
from openmdao.api import ExplicitComponent, ImplicitComponent

from rkopenmdao.om_data_exchange import OMDataExchange


@dataclass
class UnsteadyComponentMixin:
    """
    Mixin which makes a component visible to the OpenMDAO ODE implementation and
    allows data exchange from the ODE to the components inheriting from this.

    Attributes
    ----------
    om_data_exchange: OMDataExchange
        Object with which data is given into the component by the time integration.
        Will be overwritten by OpenMDAOODE.
    """

    om_data_exchange: OMDataExchange = field(default_factory=OMDataExchange, init=False)


class ExplicitUnsteadyComponent(UnsteadyComponentMixin, ExplicitComponent):
    """
    Explicit component for time integration relevants part of the problem given to
    OpenMDAOODE.

    For the documentation of methods, parameters and attributes refer to the
    documentations of the parent classes.
    """

    def __init__(self, **kwargs):
        UnsteadyComponentMixin.__init__(self)
        ExplicitComponent.__init__(self, **kwargs)


class ImplicitUnsteadyComponent(UnsteadyComponentMixin, ImplicitComponent):
    """
    Implicit component for time integration relevants part of the problem given to
    OpenMDAOODE.

    For the documentation of methods, parameters and attributes refer to the
    documentations of the parent classes.
    """

    # pylint: disable=abstract-method
    # This iis still intended to be an abstract class itself, so it is intended to not
    # overwrite `apply_nonlinear` at this point.
    def __init__(self, **kwargs):
        UnsteadyComponentMixin.__init__(self)
        ImplicitComponent.__init__(self, **kwargs)
