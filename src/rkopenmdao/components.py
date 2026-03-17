from openmdao.api import ExplicitComponent, ImplicitComponent

from rkopenmdao.om_data_exchange import OMDataExchange


class UnsteadyComponentMixin:
    om_data_exchange: OMDataExchange

    def __init__(self):
        self.om_data_exchange = OMDataExchange()


class ExplicitUnsteadyComponent(UnsteadyComponentMixin, ExplicitComponent):
    """"""

    def __init__(self, **kwargs):
        UnsteadyComponentMixin.__init__(self)
        ExplicitComponent.__init__(self, **kwargs)


class ImplicitUnsteadyComponent(UnsteadyComponentMixin, ImplicitComponent):
    """"""

    def __init__(self, **kwargs):
        UnsteadyComponentMixin.__init__(self)
        ImplicitComponent.__init__(self, **kwargs)
