## Developer notes

To build and use the Python library locally:

    mkdir build
    cd build
    cmake -DBUILD_SHARED_LIBS=OFF -DPRIMA_ENABLE_C=ON -DPRIMA_ENABLE_PYTHON=ON -DCMAKE_INSTALL_PREFIX=$(pwd)/install ..
    make -j8 install
    # There should now be a prima folder in $(pwd)/install, set the PYTHONPATH in order to use it
    export PYTHONPATH=$(pwd)/install/
    # If the above instructions worked correctly this test should pass (it simply runs the examples)
    ctest -R py
    # If you would like to test the repo cd into python/tests and run
    pytest -s .

To build a wheel that can be installed in user or system site packages:

    pipx run build

This will create a wheel in the `dist` directory that can be installed with pip. This process also runs all the tests.
For most use cases it is easier to use pipx to build and test things, but the CMake instructions were also provided in the event that
there is a failure when running pipx and the developer desires some more insight into the process.
