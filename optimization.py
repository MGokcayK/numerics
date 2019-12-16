""" 
Author  : Mehmet Gokcay Kabatas
Mail    : mgokcaykdev@gmail.com
Version : 0.1
Date    : 10/12/2019
Update  : 16/12/2019
Python  : 3.6.5

Update Note : Adding descriptions.

This script written by @Author for personal usage. 

"""

class OneD():
    """
    This class written for numerical methods for One Dimentional 
    Optimization approachs.

        @Methods :
        - GoldenSection
        - Parabolic Interpolation
        - Newton

        @Usage : 
        ...
        solver = ODE()
        solver.@Methods
        ...

    """

    def GoldenSection(self, xLow, xHigh, maxIt, func, minMax = "Max", es=1):
        """ 
        Golden Section method for optimization.
        
        Arguments :
        -------------
        xLow = initial lower start value of x.

        xHigh = initial higher start value of x.

        maxIt = maximum iteration number.

        func = target function.

        minMax = Minimization or Maximization. Default Value is "Max" mean
        maximization. For Minimization, minMax should be "Min".

        es = Accepted error value.

        Return :
        --------
            Optimum value of function.

            @Usage :
            ... 
            
            def f(x):
                return 2sinx - x**2/10

            opt = OneD()
            val = opt.GoldenSection(xLow, xHigh, maxIt, f, minMax = "Min")
            ...
        
        """
        R = (pow(5,0.5) - 1) / 2  # golden ratio
        xl = xLow ; xu = xHigh
        i = 1 # iteration 
        d = R * (xu - xl) # distance
        self.x1 = xl + d ; self.x2 = xu - d # internal location
        self.f1 = func([self.x1]) ; self.f2 = func([self.x2])
        self.minMax = minMax
        xopt, fx = self.__GoldenCond() 
        for i in range(maxIt):
            d *= R 
            if ((self.minMax == "Max") and (self.f1 > self.f2)) or ((self.minMax == "Min") and (self.f1 < self.f2)) :
                xl = self.x2
                x2 = self.x1
                self.x1 = xl + d
                self.f2 = self.f1
                self.f1 = func([self.x1])
            else:
                xu = self.x1 
                self.x1 = self.x2 
                self.x2 = xu - d
                self.f1 = self.f2
                self.f2 = func([self.x2])
            xopt, fx = self.__GoldenCond()
            if xopt != 0.:
                ea = (1. -R) * abs((xu - xl)/xopt) * 100
            if (ea < es):
                break
        return xopt

    def __GoldenCond(self):
        if ((self.minMax == "Max") and (self.f1 > self.f2)) or ((self.minMax == "Min") and (self.f1 < self.f2)) :
            xopt = self.x1 # local optimum
            fx = self.f1
        else:
            xopt = self.x2 
            fx = self.f2 
        return xopt, fx

    def ParabolicInterpolation(self, x0, x1, x2, maxIt, func):
        """ 
        Parabolic Interpolation method for optimization.
        
        Arguments :
        -------------
        x0, x1, x2 = initial guesses.

        maxIt = maximum iteration number.

        func = target function.

        Return :
        --------
            Optimum value of function.

            @Usage :
            ... 

            def f(x):
                return 2sinx - x**2/10


            opt = OneD()
            val = opt.ParabolicInterpolation(x0, x1, x2, maxIt, f)
            ...
        
        """
        args = [x0, x1, x2]
        for i in range(maxIt):
            f0 = func([args[0]])
            f1 = func([args[1]])
            f2 = func([args[2]])
            x3 = (f0 * (args[1]**2 - args[2]**2) + f1 * (args[2]**2 - args[0]**2) + f2 * (args[0]**2 - args[1]**2))
            dev = (2 * f0 * (args[1] - args[2]) + 2 * f1 * (args[2] - args[0]) + 2 * f2 * (args[0] - args[1]))
            if dev == 0:
                break
            x3 /= dev
            f3 = func([x3])
            for x in args:
                if x > x3:
                    ind = args.index(x)
                    args[2] = x
                    args[0] = args[ind-1]
                    args[1] = x3
                    break
        return args[1]

    def Newton(self, x0, maxIt, der, dder):
        """ 
        Newton method for optimization.
        
        Arguments :
        -------------
        x0 = initial guess.

        maxIt = maximum iteration number.

        der = derivative of function.

        dder = seconn derivation of function.

        Return :
        --------
            Optimum value of function.

            @Usage :
            ... 

            def f(x):
                return 2sin(x) - x^2/10

            def df(x):
                return 2cos(x) - x/5

            def ddf(x):
                retunr -2sin(x) - 1/5

            opt = OneD()
            val = opt.Newton(x0, maxIt, df, ddf)
            ...
        
        """
        for i in range(maxIt):
            xNext = x0 - (der([x0]) / dder([x0]))
            x0 = xNext  
        return xNext
