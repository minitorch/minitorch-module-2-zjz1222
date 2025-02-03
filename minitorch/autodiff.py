from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from queue import Queue
from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    
    # TODO: Implement for Task 1.1.
    vals_h, vals_l = [], []
    for index, val in enumerate(vals):
        if index == arg:
            vals_h.append(val + epsilon)
            vals_l.append(val - epsilon)
        else:
            vals_h.append(val)
            vals_l.append(val)
    
    h = f(*vals_h)
    l = f(*vals_l)
    return (h - l) / (2*epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    
    scalar_queue = Queue()
    scalar_queue.put(variable)
    topological_list = [variable]
    visit_list = []

    while not scalar_queue.empty():
        u = scalar_queue.get()
        visit_list.append(u.unique_id)
        if not u.is_leaf():
            for v in u.parents:
                if (not v.is_constant()) and (not v.unique_id in visit_list):
                    scalar_queue.put(v)    
                    topological_list.append(v)
                    visit_list.append(v.unique_id)


    return topological_list


#pytest tests/test_autodiff.py -k test_backprop1
def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    topological_list = topological_sort(variable)
    derivative = {}
    derivative[variable.unique_id] = deriv
    
    for index in range(len(topological_list)):
        u = topological_list[index]
        deriv_u = derivative.get(u.unique_id, None)
        
        if u.is_leaf():
            u.accumulate_derivative(deriv_u)

        if not u.is_leaf():
            back = u.chain_rule(deriv_u)
            back = list(back)
            for index_var in range(len(back)):
                var, deriv_v = back[index_var]

                var_derivative = derivative.get(var.unique_id, None)
                if var_derivative is None:
                    var_derivative = 0
                var_derivative += deriv_v

                derivative[var.unique_id] = var_derivative


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
