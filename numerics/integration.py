""" 
Author  : Mehmet Gokcay Kabatas
Mail    : mgokcaykdev@gmail.com
Version : 0.1
Date    : 13/12/2019
Update  : 16/12/2019
Python  : 3.6.5

Update Note : Adding descriptions.

This script written by @Author for personal usage. 

Prerequest : numpy

"""

import numpy as np 

class OneDIntegralwithFunction():
    """
    This class written for numerical methods for One Dimentional Integral
    with given Function. 

        @Methods :
        - Trapezoid
        - TrapezoidUnequalSegments
        - Simpson's 1/3 == SimpsonOneThird
        - Simpson's 3/8 == SimpsonThirdEight
        - Romberg
        - Adaptive Quadrature
        - Two Point Gauss Legendre

        @Usage : 
        ...
        integral = OneDIntegralwithFunction()
        integral.@Methods
        ...

    """
    
    def Trapezoid(self, l, u, n, Fxdx):
        """
            This function calculate Trapezoid method w.r.t func.

            Arguments :
            -------------
                l = lower boundary of integral.

                u = upper boundary of integral.

                n = number of segments.

                Fxdx = integrate function.

            Return :
            --------
                Value of integrated function.        

                @Usage :
                ...
                def f(x):
                    return x**2

                integral = OneDIntegralwithFunction()
                integral = Trapezoid(l, u, n, f)
                ...
        """
        sum = 0 
        h = (u - l) / n
        wd = u - l
        f0 = Fxdx(l)
        for i in range(1,n):
            l += h
            sum += Fxdx(l)
        I =  wd * (f0 + 2 * sum + Fxdx(u)) / (2*n)
        return I

    def TrapezoidUnequalSegments(self, x, Fxdx):
        """
            This function calculate Trapezoid method which
            is not equal segments w.r.t func.

            @Note : x should be array.

            Arguments :
            -------------
                x = x points of function.

                Fxdx = integrate function.

            Return :
            --------
                Value of integrated function.        

                @Usage :
                ...
                def f(x):
                    return x**2

                x = [0, 1, 3, 3.2, 5]

                integral = OneDIntegralwithFunction()
                integral = TrapezoidUnequalSegments(x,f)
                ...
        """
        sum = 0
        for i in range(1,len(x)):
            temp = Fxdx(i) - Fxdx(i - 1)
            temp /= 2
            sum += (x[i]-x[i-1]) * temp
        return sum
        
    def SimpsonOneThird(self, l, u, n, Fxdx):
        """
            This function calculate Simpson's 1/3 method w.r.t func.

            Arguments :
            -------------
                l = lower boundary of integral.

                u = upper boundary of integral.

                n = number of segments.

                Fxdx = integrate function.

            Return :
            --------
                Value of integrated function.        

                @Usage :
                ...
                def f(x):
                    return x**2

                integral = OneDIntegralwithFunction()
                integral = SimpsonOneThird(l, u, n, f)
                ...
        """ 
        wd = u - l
        f0 = Fxdx(l)
        f11 = 0
        f12 = 0
        for i in range(1,n):
            l += wd / n
            if (i % 2 == 0):
                f12 += Fxdx(l)
            else:
                f11 += Fxdx(l)
        if (n == 1):
            f11 = Fxdx(l + wd / 2)
            n = 2
        f2 = Fxdx(u)
        return wd * (f0 + 4 * f11 + 2 * f12 + f2) / (3*n)

    def SimpsonThirdEight(self, l, u, Fxdx):
        """
            This function calculate Simpson's 3/8 method w.r.t func.

            Arguments :
            -------------
                l = lower boundary of integral.

                u = upper boundary of integral.

                Fxdx = integrate function.

            Return :
            --------
                Value of integrated function.        

                @Usage :
                ...
                def f(x):
                    return x**2

                integral = OneDIntegralwithFunction()
                integral = SimpsonThirdEight(l, u, n, f)
                ...
        """ 
        wd = u - l
        f0 = Fxdx(l)
        f1 = Fxdx(l + wd / 3)
        f2 = Fxdx(l + wd * 2 / 3)
        f3 = Fxdx(u)
        return wd * (f0 + 3 * f1 + 3 * f2 + f3) / 8

    def Romberg(self, l, u, maxIt, Fxdx):
        """
            This function calculate Romberg method w.r.t func.

            Arguments :
            -------------
                l = lower boundary of integral.

                u = upper boundary of integral.

                maxIt = maximum iteration.

                Fxdx = integrate function.

            Return :
            --------
                Value of integrated function.        

                @Usage :
                ...
                def f(x):
                    return x**2

                integral = OneDIntegralwithFunction()
                integral = Romberg(l, u, maxIt, f)
                ...
        """ 
        I = np.empty((maxIt,maxIt))
        n = 0
        I[0][0] = self.Trapezoid(l,u,n+1,Fxdx)
        for cnt in range(maxIt):
            n = 2**cnt
            I[cnt][0] = self.Trapezoid(l,u,n,Fxdx)
            for k in range(1,cnt+1):
                j = cnt - k
                I[j][k] = (4**(k) * I[j+1][k-1]-I[j][k-1])/ (4**(k)-1)
        return I[0][-1]

    def AdaptiveQuadrature(self, l, u, Fxdx):
        """
            This function calculate Adaptive Quadrature method w.r.t func.

            Arguments :
            -------------
                l = lower boundary of integral.

                u = upper boundary of integral.

                Fxdx = integrate function.

            Return :
            --------
                Value of integrated function.        

                @Usage :
                ...
                def f(x):
                    return x**2

                integral = OneDIntegralwithFunction()
                integral = AdaptiveQuadrature(l, u, f)
                ...
        """ 
        tol = 0.000001
        h = (l + u) / 2
        fl = Fxdx(l)
        fh = Fxdx(h)
        fu = Fxdx(u)
        return self.__qstep(l,u,tol,fl,fh,fu, Fxdx)

    def __qstep(self, l, u, tol, fl, fh, fu, Fxdx):
        h1 = u - l
        h2 = h1 / 2
        h = (l + u) / 2
        fd = Fxdx((l + h) / 2)
        fe = Fxdx((h + u) / 2)
        I1 = h1 / 6 * (fl + 4 * fh + fu)
        I2 = h2 / 6 * (fl + 4 * fd + 2 * fh + 4 * fe + fu)
        if (abs(I2 - I1) < tol):
            I = I2 + (I2 - I1) / 15
        else:
            Il = self.__qstep(l,h,tol,fl,fd,fh,Fxdx)
            Iu = self.__qstep(h,u,tol,fh,fe,fu,Fxdx)
            I = Il + Iu
        return I

    def TwoPointGaussLegendre(self, Fxdx):
        """
            This function calculate Two Point Gauss Legendre method w.r.t func.

            Arguments :
            -------------

                Fxdx = integrate function.

            Return :
            --------
                Value of integrated function.        

                @Usage :
                ...
                def f(x):
                    return x**2

                integral = OneDIntegralwithFunction()
                integral = TwoPointGaussLegendre(f)
                ...
        """ 
        return (Fxdx(-1/(3**0.5)) - Fxdx(1/(3**0.5)))


class OneDIntegralwithData():
    """
    This class written for numerical methods for One Dimentional Integral
    with given data. 

        @Methods :
        - Trapezoid
        - TrapezoidUnequalSegments
        - Simpson's 1/3 == SimpsonOneThird
        - Romberg

        @Usage : 
        ...
        integral = OneDIntegralwithData()
        integral.@Methods
        ...

    """
    
    def Trapezoid(self, x, y):
        """
            This function calculate Trapezoid method w.r.t data.

            @Note : x and y should be array.

            Arguments :
            -------------
                x = x values of data.

                y = y values of data.

            Return :
            --------
                Value of integrated function.        

                @Usage :
                ...
                x = [...]
                y = [...]

                integral = OneDIntegralwithData()
                integral = Trapezoid(x,y)
                ...
        """
        sum = 0 
        wd = max(x) - min(x)
        f0 = min(x)
        f0ind = x.index(f0)
        f2 = max(x)
        f2ind = x.index(f2)
        for i in range(len(x)):
            if (i!=f0ind) and (i!=f2ind):
                sum += x[i]
        I =  wd * (f0 + 2 * sum + f2) / (2*(len(x)))
        return I

    def TrapezoidUnequalSegments(self, x, y):
        """
            This function calculate Trapezoid Unequal Segments
             method w.r.t data.

            @Note : x and y should be array.

            Arguments :
            -------------
                x = x values of data.

                y = y values of data.

            Return :
            --------
                Value of integrated function.        

                @Usage :
                ...
                x = [...]
                y = [...]

                integral = OneDIntegralwithData()
                integral = TrapezoidUnequalSegments(x,y)
                ...
        """
        sum = 0
        for i in range(1,len(x)):
            temp = y[i] - y[i - 1]
            temp /= 2
            sum += (x[i]-x[i-1]) * temp
        return sum
        
    def SimpsonOneThird(self, x, y):
        """
            This function calculate Simpson's 1/3 method w.r.t data.

            @Note : x and y should be array.

            Arguments :
            -------------
                x = x values of data.

                y = y values of data.

            Return :
            --------
                Value of integrated function.        

                @Usage :
                ...
                x = [...]
                y = [...]

                integral = OneDIntegralwithData()
                integral = SimpsonOneThird(x,y)
                ...
        """
        wd = max(x) - min(x)
        f0 = min(x)
        f0ind = x.index(f0)
        f2 = max(x)
        f2ind = x.index(f2)
        f11, f12 = 0, 0
        for i in range(len(x)):
            if (i!=f0ind) and (i!=f2ind):
                if (i % 2 == 0):
                    f12 += x[i]
                else:
                    f11 += x[i]
        return wd * (f0 + 4 * f11 + 2 * f12 + f2) / (3*len(x))

    def Romberg(self, x, y, maxIt):
        """
            This function calculate Romberg method w.r.t data.

            @Note : x and y should be array.

            Arguments :
            -------------
                x = x values of data.

                y = y values of data.

                maxIt = maximum iteration. 

            Return :
            --------
                Value of integrated function.        

                @Usage :
                ...
                x = [...]
                y = [...]

                integral = OneDIntegralwithData()
                integral = Romberg(x,y,maxIt)
                ...
        """
        I = np.empty((maxIt,maxIt))
        n = 0
        I[0][0] = self.Trapezoid(x,y)
        for cnt in range(maxIt):
            n = 2**cnt
            I[cnt][0] = self.Trapezoid(x,y)
            for k in range(1,cnt+1):
                j = cnt - k
                I[j][k] = (4**(k) * I[j+1][k-1]-I[j][k-1])/ (4**(k)-1)
        return I[0][-1]
        

class __TwoDIntegral():
    """
        Uncomplete 2D integral methods..
    """
    def __Trapezoid(self, xl, xu, nX, yl, yu, nY, Fxdx):
        h, wd = [], []
        h.append( (xu - xl) / nX )
        h.append( (yu - yl) / nY )
        wd.append( xu - xl )
        wd.append( yu - yl )
        lx,ly = xl, yl 
        I = []
        for j in range(nY+1):            
            sum = 0
            f0 = Fxdx([xl,yl])
            for i in range(1,nX):
                xl += h[0]               
                sum += Fxdx([xl,yl])
            I.append(wd[0] * (f0 + 2. * sum + Fxdx([xu,yl])) / (2.*nX))
            xl = lx
            yl += h[1]
        res = wd[1] * (I[0] + 2*I[1] + I[2]) / (2*nY)
        return res / (wd[0] * wd[1])


        
