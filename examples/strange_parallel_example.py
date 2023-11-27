import openmdao.api as om


class StrangeComponent(om.ExplicitComponent):
    def setup(self):
        self.add_input(f"x_{self.comm.rank}", shape=1, val=self.comm.rank)
        self.add_output("y", distributed=True, val=2.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["y"] = 2 * inputs[f"x_{self.comm.rank}"]

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None
    ):
        if mode == "fwd":
            d_outputs["y"] = 2 * d_inputs[f"x_{self.comm.rank}"]
        elif mode == "rev":
            d_inputs[f"x_{self.comm.rank}"] = 2 * d_outputs["y"]


if __name__ == "__main__":
    strange_prob = om.Problem()

    strange_prob.model.add_subsystem("strange_comp", StrangeComponent(), promotes=["*"])

    strange_prob.setup()

    strange_prob.run_model()

    # print(strange_prob.comm.rank, strange_prob.get_val(f"x_{strange_prob.comm.rank}"))
    # print(strange_prob.comm.rank, strange_prob.get_val("y", get_remote=False))

    strange_prob.model.list_inputs(shape=True, global_shape=True, all_procs=True)

    strange_prob.model.list_outputs(shape=True, global_shape=True, all_procs=True)
    # for i in range(strange_prob.comm.size):
    #     if i == strange_prob.comm.rank:
    #
    #     strange_prob.comm.Barrier()
