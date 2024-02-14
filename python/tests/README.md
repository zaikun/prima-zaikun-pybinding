Requirements
==============

These are high level requirements for the python package. Next to the requirement is the name of the test file and the name of the test
within that file which implements it. The testfile/testname is written in such a way that it can be passed as an argument to `pytest` in
order to run that test by itself.


## Requirements for basic functionality, algorithm autoselection, and explicit algorithm selection
- [x] providing nonlinear constraints alone calls COBYLA and the constraints are successfully applied - test_basic_functionality.py::test_provide_nonlinear_constraints_alone
- [x] providing nonlinear constraints alone and selecting COBYLA calls COBYLA and the constraints are successfully applied - test_basic_functionality.py::test_provide_nonlinear_constraints_alone_and_select_COBYLA
- [x] providing linear constraints alone calls LINCOA and the constraints are successfully applied - test_basic_functionality.py::test_provide_linear_constraints_alone
- [x] providing linear constraints alone and selecting LINCOA calls LINCOA and the constraints are successfully applied - test_basic_functionality.py::test_provide_linear_constraints_alone_and_select_LINCOA
- [x] providing bounds alone calls BOBYQA and the bounds are successfully applied - test_basic_functionality.py::test_provide_bounds_alone
- [x] providing bounds alone and selecting BOBYQA calls BOBYQA and the bounds are successfully applied - test_basic_functionality.py::test_provide_bounds_alone_and_select_BOBYQA
- [x] not providing any bounds, linear constraints, or nonlinear constraints calls NEWUOA and provides the optimal unbounded/unconstrained solution - test_basic_functionality.py::test_not_providing_bounds_linear_constraints_or_nonlinear_constraints
- [x] not providing any bounds, linear constraints, or nonlinear constraints and selecting NEWUOA calls NEWUOA and provides the optimal unbounded/unconstrained solution - test_basic_functionality.py::test_not_providing_bounds_linear_constraints_or_nonlinear_constraints_and_select_NEWUOA
- [x] not providing any bounds, linear constraints, or nonlinear constraints and selecting UOBYQA calls UOBYQA and provides the optimal unbounded/unconstrained solution - test_basic_functionality.py::test_not_providing_bounds_linear_constraints_or_nonlinear_constraints_and_select_UOBYQA
## Requirements for combining constraints and bounds
- [x] providing linear and nonlinear constraints together calls COBYLA and the constraints are successfully applied - test_combining_constraints.py::test_providing_linear_and_nonlinear_constraints
- [x] providing bounds and linear constraints together calls LINCOA and the constraints/bounds are successfully applied - test_combining_constraints.py::test_providing_bounds_and_linear_constraints
- [x] providing bounds and nonlinear constraints together calls COBYLA and the constraints/bounds are successfully applied - test_combining_constraints.py::test_providing_bounds_and_nonlinear_constraints
- [x] provoding bounds and linear and nonlinear constraints together calls COBYLA and the constraints/bounds are successfully applied - test_combining_constraints.py::test_providing_bounds_and_linear_and_nonlinear_constraints
## Requirements for data types
- [x] providing a nonlinear constraint function returning a numpy array works without warnings or errors - test_data_types.py::test_constraint_function_returns_numpy_array
- [x] providing a nonlinear constraint function returning a Python list works without warnings or errors - test_data_types.py::test_constraint_function_returns_list
- [x] providing a nonlinear constraint function returning a scalar works without warnings or errors - test_data_types.py::test_constraint_function_returns_scalar
- [x] providing x0 as numpy array works without warnings or errors - test_data_types.py::test_x0_as_list
- [x] providing x0 as Python list works without warnings or errors - test_data_types.py::test_x0_as_array
- [x] providing x0 as scalar works without warnings or errors - test_data_types.py::test_x0_as_scalar
## Requirements for various options
- [ ] various options can be successfully provided
  - [x] ftarget - test_options.py::test_ftarget
  - [x] iprint - test_options.py::test_iprint
  - [x] max function evaluations via maxfun (name used by SciPy) - test_options.py::test_maxfun
  - [x] max function evaluations via maxfev (name used by PRIMA) - test_options.py::test_maxfev
  - [ ] npt
  - [ ] rhobeg
  - [ ] rhoend
- [x] an objective function without args can be used sucessfully - test_options.py::test_normal_function
- [x] an objective function with args can be used successfully - test_options.py::test_function_with_regular_args
- [x] an objective function with *args can be used successfully - test_options.py::test_function_with_star_args
- [x] providing a callback leads to the callback being successfully called - test_options.py::test_callback
- [x] providing a callback that returns True leads to early termination - test_options.py::test_callback_early_termination
- [x] providing anonymous lambda as: objective function, constraint function, callback works without warnings or errors - test_anonymous_lambda.py::test_anonymous_lambda

## Requirements for compatibility with existing APIs
- [x] compatible with scipy.optimize.minimize API - test_compatible_interface.py
- [ ] compatible with PDFO API - test_compatible_interface.py (needs remaining algorithms)
## Requirements for processing of lists of constraints
- [ ] providing a list of nonlinear constraint functions without providing either the total dimension or the dimension of each function successfully determines the total number of contraints
- [ ] providing a list mixing linear and nonlinear constraints leads to successfully setting the appropraite constraints and their successful application
- [ ] providing a list of linear constraints leads to them being successfully combined and applied
## Requirements for regression tests
These tests are for behavior observed during testing that we want to make sure remains fixed.
- [x] calling with a method that is not available throws an exception and exits cleanly (i.e. no other warnings or error or hanging of the interpreter) (relates to the test regarding anonymous lambda functions) - test_anonymous_lambda.py::test_anonymous_lambda_unclean_exit
## Requirements for ordering a lizard
This tests calling PRIMA with unexpected options and verifying that it behaves reasonably. The name comes from an old joke about a QA engineer:

    A QA engineer walks into a bar.
    He orders a beer.
    He orders 2 beers.
    He orders 0 beers, -1 beers, he orders a lizard.
    Satisfied that the bartender is performing his job adequately, he prepares to sign off on the release.
    A customers walks in and asks where the bathroom is. The whole bar erupts in flames.

- [ ] Calling COBYLA without nonlinear constraints results in a warning informing the user they may have selected an inappropriate algorithm
- [ ] Calling LINCOA without linear or nonlinear constraints results in a warning informing the user they may have selected an inappropriate algorithm
- [ ] Calling LINCOA with nonlinear constraints results in a warning informing the user that the nonlinear constrains will be ignored
- [ ] Calling BOBYQA without bounds or linear or nonlinear constraints results in a warming informing the user they may have selected an inappropraite algorithm
- [ ] Calling BOBYQA with linear constraints results in a warning informing the user that the linear constraints will be ignored
- [ ] Calling BOBYQA with nonlinear constraints results in a warning informing the user that the nonlinear constraints will be ignored
- [ ] Calling BOBYQA with both linear and nonlinear constraints results in the same behavior as calling BOBYQA with nonlinear constraints
- [ ] Calling UOBYQA with bounds results in a warning informing the user that the bounds will be ignored
- [ ] Calling UOBYQA with linear constriants results in a warning informing the user that the linear constriants will be ignored
- [ ] Calling UOBYQA with nonlinear constriants results in a warning informing the user that the nonlinear constriants will be ignored
- [ ] Calling NEWUOA with bounds results in a warning informing the user that the bounds will be ignored
- [ ] Calling NEWUOA with linear constriants results in a warning informing the user that the linear constriants will be ignored
- [ ] Calling NEWUOA with nonlinear constriants results in a warning informing the user that the nonlinear constriants will be ignored
