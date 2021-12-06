"""
how to run:
- Linux: python main.py
- Windows: python3 main.py
"""
import numpy

DIMENSION = 20


def init_a(gamma):
    """
    initialize matrix A
     γ -1  0 ...  0 0
    -1  γ -1 ...  0 0
        ...    ...
     0  0  0 ... -1 γ
    """
    a = [[numpy.float64(0) for i in range(DIMENSION)] for j in range(DIMENSION)]
    for i in range(0, DIMENSION):
        for j in range(0, DIMENSION):
            if i == j:
                a[i][j] = numpy.float64(gamma)
            if i == j + 1 or j == i + 1:
                a[i][j] = numpy.float64(-1)
    return a


def init_b(gamma):
    """
    initialize b vector
    b = (γ - 1, γ - 2, ..., γ - 2, γ - 1)
    """
    b = [numpy.float64(gamma - 2) for i in range(DIMENSION)]
    b[0] = numpy.float64(gamma - 1)
    b[DIMENSION - 1] = numpy.float64(gamma - 1)
    return b


def approx(a_matrix, b_vector, x):
    """
    compute the stopping criteria
    ||Ax - b|| / ||b||
    x = approximate solution
    using euclidean norm: ||x||_2 = (sum(x_i)^2)^1/2
    """
    dividend = numpy.float64(0)
    divisor = numpy.float64(0)
    # Ax - b
    ax = numpy.matmul(a_matrix, x, dtype=numpy.float64)
    axb = numpy.subtract(ax, b_vector, dtype=numpy.float64)
    # sum
    for i in range(0, DIMENSION):
        dividend = dividend + numpy.power(axb[i], 2, dtype=numpy.float64)
        divisor = divisor + numpy.power(b_vector[i], 2, dtype=numpy.float64)
    # ^1/2
    dividend = numpy.sqrt(dividend)
    divisor = numpy.sqrt(divisor)
    return numpy.divide(dividend, divisor)


def jacobi(a_matrix, b_vector, x_0, tolerance, max_iterations):
    """
    jacobi method
    returns K number of necessary iterations
    if stopping criteria isn't met in max_iterations steps, return "Result not found"
    """
    k = 1
    x = []
    while k <= max_iterations:
        x = jacobi_x(a_matrix, b_vector, x_0)
        if x is None:
            return "Diverges"
        if numpy.less(approx(a_matrix, b_vector, x), tolerance):
            return k
        k = k + 1
        x_0 = x
    return "Result not found"


def jacobi_x(a_matrix, b_vector, x_0):
    """
    computing of x vector in jacobi method
    Q^-1((Q - A)x_0 + b)
    Q = D = non-zero values only on diagonal, diagonal same as matrix A
    """
    q = [[numpy.float64(0) for i in range(DIMENSION)] for j in range(DIMENSION)]
    for i in range(0, DIMENSION):
        q[i][i] = numpy.float64(a_matrix[i][i])
    q_m1 = numpy.linalg.inv(q)
    if not converges(q, a_matrix):
        return None
    return numpy.matmul(q_m1, numpy.add(numpy.matmul(numpy.subtract(q, a_matrix), x_0), b_vector))


def gauss_seidel(a_matrix, b_vector, x_0, tolerance, max_iterations):
    """
    gauss-seidel method
    returns K number of necessary iterations
    if stopping criteria isn't met in max_iterations steps, return "Result not found"
    """
    k = 1
    x = []
    while k <= max_iterations:
        x = gs_x(a_matrix, b_vector, x_0)
        if x is None:
            return "Diverges"
        if numpy.less(approx(a_matrix, b_vector, x), tolerance):
            return k
        k = k + 1
        x_0 = x
    return "Result not found"


def gs_x(a_matrix, b_vector, x_0):
    """
    computing of x vector in gauss-seidel method
    Q = D + L
    A = D + L + U
    x = Q^-1 * (-Ux_0 + b)
    """
    el = [[numpy.float64(0) for i in range(DIMENSION)] for j in range(DIMENSION)]
    for i in range(0, DIMENSION):
        for j in range(0, DIMENSION):
            if i == j + 1:
                el[i][j] = numpy.float64(-1)
    d = [[numpy.float64(0) for i in range(DIMENSION)] for j in range(DIMENSION)]
    for i in range(0, DIMENSION):
        d[i][i] = numpy.float64(a_matrix[i][i])
    q = numpy.add(d, el)
    u = numpy.subtract(a_matrix, q)
    q_m1 = numpy.linalg.inv(q)
    if not converges(q, a_matrix):
        return None
    return numpy.matmul(q_m1, numpy.add(numpy.matmul(u*-1, x_0), b_vector))


def converges(q, a_matrix):
    """
    W = E - Q^-1*A
    ρ(W) := max{|λ|: λ being an eigenvalue of W}
    return True if ρ(W) < 1
    return False otherwise (diverges)
    """
    w = numpy.subtract(numpy.identity(20), numpy.matmul(numpy.linalg.inv(q), a_matrix))
    eigenvalues = numpy.linalg.eig(w)[0]
    rho = max(eigenvalues.min(), eigenvalues.max(), key=abs)
    if numpy.less(rho, numpy.float64(1)):
        return True
    else:
        return False


if __name__ == '__main__':
    gammas = {5, 2, 0.5}
    tol = numpy.float64(pow(10, -6))
    x0 = [numpy.float64(0) for i in range(DIMENSION)]
    max_iter = 1000
    for index, value in enumerate(gammas):
        print("γ = " + str(value))
        am = init_a(value)
        bv = init_b(value)
        jac = jacobi(am, bv, x0, tol, max_iter)
        print("Jacobi: " + str(jac))
        gs = gauss_seidel(am, bv, x0, tol, max_iter)
        print("Gauss-Seidel: " + str(gs))
        print("*"*40)

