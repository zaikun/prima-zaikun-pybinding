Things to test
==============

- [x] objective function without args - test_fun_with_args.py
- [x] objective function with args - test_fun_with_args.py
- [ ] callback
- [x] bounds - test_bounds.py
- [ ] linear constraints
- [ ] nonlinear constraints
- [ ] linear and nonlinear constraints
- [ ] bounds and linear constraints
- [ ] bounds and nonlinear constraints
- [ ] bounds and linear and nonlinear constraints
- [ ] constraint function returning numpy array
- [ ] constraint function returning list
- [ ] auto-selection of algorithm
- [ ] options
- [x] compatibility with scipy.optimize.minimize API - test_compatible_interface.py
- [ ] compatibility with PDFO API - test_compatible_interface.py (needs remaining algorithms)
- [x] providing x0 as either a Python list, a numpy array, or a scalar - test_x0_list_array_scalar.py
- [x] test anonymous lambda as: objective function, constraint function, callback - test_anonymous_lambda.py
- [x] test calling with an improper method throws an exception and can still exit cleanly (relates to the test regarding anonymous lambda functions) - test_anonymous_lambda.py
- [ ] cobyla with: number of nonlinear constraints provided, not provided, and with a list of nonlinear constraints only some of which provide the number
- [ ] test each method (while most will be tested by the above, I'd like to explicitly make sure each method is triggered)


Note that in some cases we run the exact same test with different names. While we could consolidate the test, we prefer
to have individual tests for individual pieces of functionality so that they can provide a clue as to what functionality is failing.