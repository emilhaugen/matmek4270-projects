"""
In this module we study the vibration equation

    u'' + w^2 u = f, t in [0, T]

where w is a constant and f(t) is a source term assumed to be 0.
We use various boundary conditions.

"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.sparse 

t = sp.Symbol('t')

class VibSolver:
    """
    Solve vibration equation::

        u'' + w**2 u = f,

    """
    def __init__(self, Nt, T, w=0.35, I=1):
        """
        Parameters
        ----------
        Nt : int
            Number of time steps
        T : float
            End time
        I, w : float, optional
            Model parameters
        """
        self.I = I
        self.w = w
        self.T = T
        self.set_mesh(Nt)

    def set_mesh(self, Nt):
        """Create mesh of chose size

        Parameters
        ----------
        Nt : int
            Number of time steps
        """
        self.Nt = Nt
        self.dt = self.T/Nt
        self.t = np.linspace(0, self.T, Nt+1)

    def ue(self):
        """Return exact solution as sympy function
        """
        return self.I*sp.cos(self.w*t)

    def u_exact(self):
        """Exact solution of the vibration equation

        Returns
        -------
        ue : array_like
            The solution at times n*dt
        """
        return sp.lambdify(t, self.ue())(self.t)

    def l2_error(self):
        """Compute the l2 error norm of solver

        Returns
        -------
        float
            The l2 error norm
        """
        u = self()
        ue = self.u_exact()
        return np.sqrt(self.dt*np.sum((ue-u)**2))

    def convergence_rates(self, m=4, N0=32):# m = 4
        """
        Compute convergence rate

        Parameters
        ----------
        m : int
            The number of mesh sizes used
        N0 : int
            Initial mesh size

        Returns
        -------
        r : array_like
            The m-1 computed orders
        E : array_like
            The m computed errors
        dt : array_like
            The m time step sizes
        """
        E = []
        dt = []
        self.set_mesh(N0) # Set initial size of mesh
        for m in range(m):
            self.set_mesh(self.Nt+10)
            E.append(self.l2_error())
            dt.append(self.dt)
        r = [np.log(E[i-1]/E[i])/np.log(dt[i-1]/dt[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(dt)

    def test_order(self, m=5, N0=100, tol=0.1):
        r, E, dt = self.convergence_rates(m, N0)
        print(r, self.order)
        assert np.allclose(np.array(r), self.order, atol=tol)

class VibHPL(VibSolver):
    """
    Second order accurate recursive solver

    Boundary conditions u(0)=I and u'(0)=0
    """
    order = 2 # setting self.order

    def __call__(self):
        u = np.zeros(self.Nt+1)
        u[0] = self.I
        u[1] = u[0] - 0.5*self.dt**2*self.w**2*u[0]
        for n in range(1, self.Nt):
            u[n+1] = 2*u[n] - u[n-1] - self.dt**2*self.w**2*u[n]
        return u

class VibFD2(VibSolver):
    """
    Second order accurate solver using boundary conditions::

        u(0)=I and u(T)=I

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """
    order = 2

    def __init__(self, Nt, T, w=0.35, I=1, f=None):
        VibSolver.__init__(self, Nt, T, w, I)
        T = T * w / np.pi

        assert T.is_integer() and T % 2 == 0

    def __call__(self):
        u = np.zeros(self.Nt+1)

        C = (2 - self.w**2 * self.dt**2)
        A = scipy.sparse.diags([np.ones(self.Nt), np.full(self.Nt+1, -C), np.ones(self.Nt)], offsets=(-1, 0, 1), format='lil')
        A[0, :2] = [1, 0]
        A[-1, -2:] = [0, 1]
        b = np.zeros_like(u)
        b[0] = self.I
        b[-1] = self.I 
        u = scipy.sparse.linalg.spsolve(A.tocsr(), b)
        return u


class VibFD2_manufactured_solution(VibFD2):
    
    def __init__(self, ue, f, Nt, T, w=0.35, I=1):
        VibSolver.__init__(self, Nt, T, w, I)
        self.ue = ue # sympy function giving exact solution 
        self.f = f 
        self.boundary_conditions = self.u_exact()[::self.Nt]


    def __call__(self):
        u = np.zeros(self.Nt+1)

        C = (2 - self.w**2 * self.dt**2)
        A = scipy.sparse.diags([np.ones(self.Nt), np.full(self.Nt+1, -C), np.ones(self.Nt)], offsets=(-1, 0, 1), format='lil')
        A[0, :2] = [1, 0]
        A[-1, -2:] = [0, 1]
        
        b = self.f(self.t) * self.dt**2
        b[0] = self.boundary_conditions[0]
        b[-1] = self.boundary_conditions[-1]

        u = scipy.sparse.linalg.spsolve(A.tocsr(), b)
        return u


class VibFD3(VibSolver):
    """
    Second order accurate solver using mixed Dirichlet and Neumann boundary
    conditions::

        u(0)=I and u'(T)=0

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """
    order = 2

    def __init__(self, Nt, T, w=0.35, I=1):
        VibSolver.__init__(self, Nt, T, w, I)
        T = T * w / np.pi
        assert T.is_integer() and T % 2 == 0

    def __call__(self):
        u = np.zeros(self.Nt+1)

        C = (2 - self.w**2 * self.dt**2)
        A = scipy.sparse.diags([np.ones(self.Nt), np.ones(self.Nt+1)*(-C), np.ones(self.Nt)], offsets=(-1, 0, 1), format='lil')
        A[0, :2] = [1, 0]
        A[-1, -2:] = [1, -C/2]
        b = np.zeros_like(u)
        b[0] = self.I
        u = scipy.sparse.linalg.spsolve(A.tocsr(), b)
        return u


class VibFD4(VibFD2):
    """
    Fourth order accurate solver using boundary conditions::

        u(0)=I and u(T)=I

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """
    order = 4

    def __call__(self):
        u = np.zeros(self.Nt+1)

        C = 30 - 12 * self.dt**2 * self.w**2 
        D = 15 - 12 * self.dt**2 * self.w**2 

        C = np.full(self.Nt+1, C)
        sixteens = np.full(self.Nt, 16)
        ones = np.ones(self.Nt-1) 

        A = scipy.sparse.diags([-ones, sixteens, -C, sixteens, -ones], offsets=(-2, -1, 0, 1, 2), format='lil')

        A[1, :6] = [10, -D, -4, 14, -6, 1]
        A[-2, -6:] = [1, -6, 14, -4, -D, 10]
        A[0, :3] = [1, 0, 0]
        A[-1, -3:] = [0, 0, 1]
        
        b = np.zeros_like(u)
        b[0] = self.I
        b[-1] = self.I 
        
        u = scipy.sparse.linalg.spsolve(A.tocsr(), b)
        return u


def test_manufactured_solution():
    w = 3
    T = 10
    Nt = 20
    
    ue = lambda : t**4
    f = lambda x : 12*x**2 + w**2 * x**4 
    I = f(0)
    S_t4 = VibFD2_manufactured_solution(ue, f, Nt=Nt, T=T, w=w, I=I)
    S_t4.test_order()

    ue = lambda : sp.exp(sp.sin(t))
    f = lambda x : np.exp(np.sin(x)) * (np.cos(x)**2 - np.sin(x) + w**2)
    I = f(0)
    S_expsin = VibFD2_manufactured_solution(ue, f, Nt=Nt, T=T, w=w, I=I)
    S_expsin.test_order()

    plt.plot(S_expsin.t, S_expsin.u_exact(), label='exact', marker='o')
    plt.plot(S_expsin.t, S_expsin(), label='approximation', marker='x')
    plt.legend()
    plt.show()


def test_order():
    w = 1
    I = 1
    T = 2*np.pi/w # one period is 2*pi/w
    Nt = 50
    #VibHPL(8, 2*np.pi/w, w).test_order() 
    #VibFD2(Nt=8, T=2*np.pi/w, w=w, I=I).test_order()
    #VibFD3(8, 2*np.pi/w, w).test_order()
    S = VibFD4(Nt=Nt, T=T, w=w, I=I)

    plt.plot(S.t, S.u_exact(), label='exact', marker='o')
    plt.plot(S.t, S(), label='approximation', marker='x')
    plt.legend()
    plt.show()

    S.test_order(N0=Nt) # fails with N0 >= 48

if __name__ == '__main__':
    test_order()
    #test_manufactured_solution()
