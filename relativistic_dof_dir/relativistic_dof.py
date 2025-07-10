from relativistic_dof_table import RelativisticDOFTable
from relativistic_dof_fit import RelativisticDOFFitModel
from relativistic_dof_lattice import RelativisticDOFLattice

class RelativisticDOFRegistry:
    """
    A registry to dynamically select a relativistic degrees of freedom calculation method.
    """
    methods = {
        "table": RelativisticDOFTable,
        "fit": RelativisticDOFFitModel,
        "lattice": RelativisticDOFLattice
    }

    @staticmethod
    def get_method(name="fit"):
        """
        Retrieves the appropriate relativistic degrees of freedom model.

        Parameters:
            name (str): The name of the method. Options: "table", "fit", "lattice".

        Returns:
            An instance of the corresponding model class.
        """
        if name not in RelativisticDOFRegistry.methods:
            raise ValueError(f"Unknown method: {name}. Choose from: {list(RelativisticDOFRegistry.methods.keys())}")

        return RelativisticDOFRegistry.methods[name]()