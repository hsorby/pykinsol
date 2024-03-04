import math

from sundials import cvode, cvode_ls, sundials_context, nvector_serial, sunmatrix_dense, sunlinsol_dense, sundials_matrix, sundials_types

from model_ode import create_states_array, create_variables_array, initialise_variables, compute_variables, compute_rates, compute_computed_constants, STATE_COUNT


def print_information():
    print("information.")


def print_headers():
    print("headers.")


def print_values(voi, states, variables):
    print(voi, states, variables)

# typedef struct {
#     void (*computeRates)(double, double *, double *, double *);
#
#     double *variables;
# } UserOdeData;


def func(_voi, _y, _y_dot, _user_data):
    y_ = nvector_serial.N_VConvertArray_Serial(_y)
    y_dot_ = nvector_serial.N_VConvertArray_Serial(_y_dot)

    _user_data["func"](_voi, y_, y_dot_, _user_data["variables"])

    nvector_serial.N_VUpdate_Serial(_y, y_)
    nvector_serial.N_VUpdate_Serial(_y_dot, y_dot_)

    return 0


def main():
    # Some information about the model.
    print_information()

    # Create our various arrays.

    voi = 0.0
    states = create_states_array()
    rates = create_states_array()
    variables = create_variables_array()

    # Initialise our states and variables, and compute our computed constants, and output the initial value/guess of our states and variables.

    initialise_variables(states, variables)
    compute_computed_constants(variables)

    print_headers()

    if not False:
        print_values(voi, states, variables)

    # Create our SUNDIALS context.

    context_ptr = sundials_context.SUNContext_Define()
    sundials_context.SUNContext_Create(None, context_ptr)

    context = sundials_context.SUNContext_Context(context_ptr)

    # Create our CVODE solver.

    solver = cvode.CVodeCreate(cvode.CV_BDF, context)

    # Initialise our CVODE solver.

    y = nvector_serial.N_VMake_Serial(STATE_COUNT, states, context)

    cvode.CVodeInit(solver, func, voi, y)

    # Set our user data.

    userData = {'func': compute_rates, 'variables': variables}

    cvode.CVodeSetUserData(solver, userData)

    # Set our maximum number of steps.

    cvode.CVodeSetMaxNumSteps(solver, 99999)

    cvode.CVodeSetMaxStep(solver, 0)

    cvode.CVodeSetMaxNumSteps(solver, 500)

    # Set our linear solver.

    matrix = sunmatrix_dense.SUNDenseMatrix(STATE_COUNT, STATE_COUNT, context)

    linearSolver = sunlinsol_dense.SUNLinSol_Dense(y, matrix, context)

    cvode_ls.CVodeSetLinearSolver(solver, linearSolver, matrix)

    # Set our relative and absolute tolerances.

    cvode.CVodeSStolerances(solver, 1.0e-12, 1.0e-12)

    # Run our model.

    # std::vector<double> outputPoints = {}

    # if (!false) {
    #     size_t i = 0
    #     double voiMax = 3.0
    #     double voiInterval = 0.00001
    #
    #     do {
    #         voi = ++i * voiInterval
    #
    #         outputPoints.push_back(voi)
    #     } while (voi < voiMax)
    #
    #     voi = 0.0
    # }

    output_points = range(0, 3000)
    output_points = [4.0e-6, 4.0e-5, 4.0e-4, 4.0e-3, 4.0e-2, 4.0e-1, 4.0e0, 4.0e1, 4.0e2, 4.0e3, 4.0e4, 4.0e5]
    for index in output_points:
        output_point = index  # / 1000.0

        # Integrate our model.
        status, voi_real = cvode.CVode(solver, output_point, y, voi, cvode.CV_NORMAL)
        voi = sundials_types.SUNDIALS_AsDouble(voi_real)

        states = nvector_serial.N_VConvertArray_Serial(y)
        compute_variables(voi, states, rates, variables)

        print_values(voi, states, variables)

    # Clean up after ourselves.

    # SUNLinSolFree(linearSolver)
    # sundials_matrix.SUNMatDestroy(matrix)
    # N_VDestroy_Serial(y)
    # CVodeFree(&solver)
    sundials_context.SUNContext_Free(context_ptr)

    # deleteArray(states)
    # deleteArray(rates)
    # deleteArray(variables)

    # return 0
    print("done.")


if __name__ == "__main__":
    main()
