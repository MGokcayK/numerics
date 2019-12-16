""" 
Author  : Mehmet Gokcay Kabatas
Mail    : mgokcaykdev@gmail.com
Version : 0.1
Date    : 04/12/2019
Update  : 14/12/2019
Python  : 3.6.5

Update Note : Arranging system of ODE methods and descriptions.

This script written by @Author for personal usage. 

Prerequest : numpy

"""
import numpy as np           

class ODE():
    """
    This class written for numerical methods for Ordinary
    Differential Equations(ODE). 

        @Methods :
        - Euler
        - Heun
        - Midpoint
        - RK2
        - RK3
        - RK4
        - RK5
        - System of ODE's Euler
        - System of ODE's RK4

        @Usage : 
        ...
        solver = ODE()
        solver.@Methods
        ...

    """
    def Euler(self, xi, xf, yi, h, dydx):
        """ Euler Method for ODE.
        
        Arguments :
        -------------
        xi = Initial value of x.

        xf = Final value of x.

        yi = Initial value of y.

        h  = Step size.

        dydx : Target function's derivative function
        which argument depend on 'x and y'.


            @ Example :
            def df(x,y):
                return (2 x + y)
            
            ... 
            solver = ODE()
            solver.Euler(0,5,2,0.2,df)
            ...

        Return :
        --------

        x_arr, y_arr : Array of x and y point(s).
        
        """
        x_arr, y_arr = [xi], [yi]
        while (xi + h <= xf):
            if (xi + h ) > xf:
                h = xf - xi
            y_next = yi + dydx(xi,yi) * h
            xi += h
            yi = y_next
            x_arr.append(xi)
            y_arr.append(yi)
        return x_arr, y_arr

    def SystemEuler(self, xi, xf, yi, h, dydx):
        """ Euler Method for System of ODE.

            @Note : yi and dydx should be array.
            
        `Derivative functions parameter should be written
        w.r.t args. Description in '@Args'.`

        
        Arguments :
        -------------
        xi = Initial value of x for each function.

        xf = Final value of x for each function.

        yi = Initial value of y for each function.

        h  = Step size.

        dydx : Target functions's derivative function
        which argument depend on args.

            @Args :
            Order of parameter of function should be same. \n
            If f1(x,y1,y2,...) and f2(x,y1,y2,...) then function's arguments should be in array args = [x,y1,y2,...]. \n

            @ Example :
            dy1dx : -0.5x + y1
            dy2dx : 0.2y1 + 0.6y2 - 3x

            : First function x parameter (x) in args[0] and y 
            parameter (y1) in args[1]. \n
            : Second function y 
            parameter (y2) in args[2].

            def df1(args):
                return (-0.5 args[0] + args[1])
            
            def df2(args):
                return (0.2 args[1] + 0.6 args[2] - 3 args[0])

            ... 
            solver = ODE()
            solver.SystemEuler(0,5,[2,2],0.2,[df1,df2])
            ...

        Return :
        --------

        x_arr, y_arr : Array of x and y point(s).
        
        """
        x_arr, y_arr, args = np.array([xi]), np.array([yi]), []
        while (xi + h <= xf):
            if (xi + h ) > xf:
                h = xf - xi
            args.append(xi)
            for g in range(len(dydx)):
                args.append(yi[g])
            for j in range(len(dydx)):
                yi[j] = yi[j] + dydx[j](args) * h
            xi += h
            x_arr = np.append(x_arr,[xi],0)
            y_arr = np.append(y_arr,[yi],0)
            args = []
        return x_arr, y_arr                

    def Heun(self, xi, xf, yi, h, dydx):
        """ Heun Method for ODE.
        
        Arguments :
        -------------
        xi = Initial value of x.

        xf = Final value of x.

        yi = Initial value of y.

        h  = Step size.

        dydx : Target function's derivative function
        which argument depend on 'x and y'.


            @ Example :
            def df(x,y):
                return (2 x + y)
            
            ... 
            solver = ODE()
            solver.Heun(0,5,2,0.2,df)
            ...

        Return :
        --------

        x_arr, y_arr : Array of x and y point(s).
        
        """
        x_arr, y_arr = [xi], [yi]
        while (xi + h <= xf):
            if (xi + h ) > xf:
                h = xf - xi
            y_next_0 = yi + dydx(xi,yi) * h
            y_next_1 = dydx(xi + h, y_next_0)
            yi = yi + (dydx(xi,yi) + y_next_1) / 2 * h
            xi += h
            x_arr.append(xi)
            y_arr.append(yi)
        return x_arr, y_arr

    def Midpoint(self, xi, xf, yi, h, dydx):
        """ Midpoint Method for ODE.
        
        Arguments :
        -------------
        xi = Initial value of x.

        xf = Final value of x.

        yi = Initial value of y.

        h  = Step size.

        dydx : Target function's derivative function
        which argument depend on 'x and y'.


            @ Example :
            def df(x,y):
                return (2 x + y)
            
            ... 
            solver = ODE()
            solver.Midpoint(0,5,2,0.2,df)
            ...

        Return :
        --------

        x_arr, y_arr : Array of x and y point(s).
        
        """
        x_arr, y_arr = [xi], [yi]
        while (xi + h <= xf):
            if (xi + h ) > xf:
                h = xf - xi
            y_next_hl = yi + dydx(xi,yi) * h / 2
            yi = yi + dydx(xi + h/2, y_next_hl) * h
            xi += h
            x_arr.append(xi)
            y_arr.append(yi)
        return x_arr, y_arr

    def RK2(self, xi, xf, yi, h, a1, a2, p1, q11, dydx):
        """ Second Order Runge Kutta Method for ODE.
        
        Arguments :
        -------------
        xi = Initial value of x.

        xf = Final value of x.

        yi = Initial value of y.

        h  = Step size.

        a1, a2, p1, q11 = Calculation constants.
        
            @Prop:
            a1 + a2 = 1
            a2 . p1 = 1/2
            a2 . q11 = 1/2

        dydx : Target function's derivative function
        which argument depend on 'x and y'.


            @ Example :
            def df(x,y):
                return (2 x + y)
            
            ... 
            solver = ODE()
            solver.RK2(0,5,2,0.2,1/2,1/2,1,1,df)
            ...

        Return :
        --------

        x_arr, y_arr : Array of x and y point(s).
        
        """
        x_arr, y_arr = [xi], [yi]
        while (xi + h <= xf):
            if (xi + h ) > xf:
                h = xf - xi
            k1 = dydx(xi, yi)
            k2 = dydx(xi + p1 * h, yi + q11 * k1 * h)
            yi = yi + (a1*k1 + a2*k2)*h
            xi += h
            x_arr.append(xi)
            y_arr.append(yi)
        return x_arr, y_arr
    
    def RK3(self, xi, xf, yi, h, dydx):
        """ Third Order Runge Kutta Method for ODE.
        
        Arguments :
        -------------
        xi = Initial value of x.

        xf = Final value of x.

        yi = Initial value of y.

        h  = Step size.

        dydx : Target function's derivative function
        which argument depend on 'x and y'.


            @ Example :
            def df(x,y):
                return (2 x + y)
            
            ... 
            solver = ODE()
            solver.RK3(0,5,2,0.2,df)
            ...

        Return :
        --------

        x_arr, y_arr : Array of x and y point(s).
        
        """
        x_arr, y_arr = [xi], [yi]
        while (xi + h <= xf):
            if (xi + h ) > xf:
                h = xf - xi
            k1 = dydx(xi, yi)
            k2 = dydx(xi + 1/2 * h, yi + 1/2 * k1 * h)
            k3 = dydx(xi + h, yi - k1*h + 2*k2*h)
            yi = yi + 1/6 * (k1 + 4*k2 + k3)*h
            xi += h
            x_arr.append(xi)
            y_arr.append(yi)
        return x_arr, y_arr

    def RK4(self, xi, xf, yi, h, dydx):
        """ Fourth Order Runge Kutta Method for ODE.
        
        Arguments :
        -------------
        xi = Initial value of x.

        xf = Final value of x.

        yi = Initial value of y.

        h  = Step size.

        dydx : Target function's derivative function
        which argument depend on 'x and y'.


            @ Example :
            def df(x,y):
                return (2 x + y)
            
            ... 
            solver = ODE()
            solver.RK4(0,5,2,0.2,df)
            ...

        Return :
        --------

        x_arr, y_arr : Array of x and y point(s).
        
        """
        x_arr, y_arr = [xi], [yi]
        while (xi + h <= xf):
            if (xi + h ) > xf:
                h = xf - xi
            k1 = dydx(xi, yi)
            k2 = dydx(xi + 1/2 * h, yi + 1/2 * k1 * h)
            k3 = dydx(xi + 1/2 * h, yi + 1/2 * k2 * h)
            k4 = dydx(xi + h , yi + k3 * h)
            yi = yi + 1/6 * (k1 + 2*k2 + 2*k3 + k4)*h
            xi += h
            x_arr.append(xi)
            y_arr.append(yi)
        return x_arr, y_arr

    def SystemRK4(self, xi, xf, yi, h, dydx):   
        """ Forth Order Runge Kutta Method for System of ODE.
            
            @Note : yi and dydx should be array.
            
        `Derivative functions parameter should be written
        w.r.t args. Description in '@Args'.`

        Arguments :
        -------------
        xi = Initial value of x.

        xf = Final value of x.

        yi = Initial value of y.

        h  = Step size.

        dydx : Target function's derivative function
        which argument depend on 'args'.

            @Args :
            Order of parameter of function should be same. \n
            If f1(x,y1,y2,...) and f2(x,y1,y2,...) then function's arguments should be in array args = [x,y1,y2,...]. \n

            @ Example :
            dy1dx : -0.5x + y1
            dy2dx : 0.2y1 + 0.6y2 - 3x

            : First function x parameter (x) in args[0] and y 
            parameter (y1) in args[1]. \n
            : Second function y 
            parameter (y2) in args[2].

            def df1(args):
                return (-0.5 args[0] + args[1])
            
            def df2(args):
                return (0.2 args[1] + 0.6 args[2] - 3 args[0])

            ... 
            solver = ODE()
            solver.SystemRK4(0,5,[2,2],0.2,[df1,df2])
            ...

        Return :
        --------

        x_arr, y_arr : Array of x and y point(s).
        
        """
        x_arr, y_arr, args = np.array([xi]), np.array([yi]), []
        k_arr = np.empty((4,len(dydx)))
        while (xi + h <= xf):
            if (xi + h ) > xf:
                h = xf - xi
            args.append(xi)
            for g in range(len(dydx)):
                args.append(yi[g])
            for i in range(len(dydx)):
                k_arr[0][i] = dydx[i](args)       
            args[0] = xi + 1/2 * h
            for i in range(len(dydx)):                
                args[i+1] = yi[i] + 1/2 * k_arr[0][i] * h
                k_arr[1][i] = dydx[i](args)
            args[0] = xi + 1/2 * h
            for i in range(len(dydx)):
                args[i+1] = yi[i] + 1/2 * k_arr[1][i] * h
                k_arr[2][i] = dydx[i](args)
            args[0] = xi + h
            for i in range(len(dydx)):
                args[i+1] = yi[i] + k_arr[2][i] * h
                k_arr[3][i] = dydx[i](args)
                yi[i] = yi[i] + 1/6 * (k_arr[0][i] + 2*k_arr[1][i] + 2*k_arr[2][i] + k_arr[3][i])*h
            xi += h         
            x_arr = np.append(x_arr,[xi],0)
            y_arr = np.append(y_arr,[yi],0)
            args = []
        return x_arr, y_arr

    def RK5(self, xi, xf, yi, h, dydx):
        """ Fifth Order Runge Kutta Method for ODE.
        
        Arguments :
        -------------
        xi = Initial value of x.

        xf = Final value of x.

        yi = Initial value of y.

        h  = Step size.

        dydx : Target function's derivative function
        which argument depend on 'x and y'.


            @ Example :
            def df(x,y):
                return (2 x + y)
            
            ... 
            solver = ODE()
            solver.RK5(0,5,2,0.2,df)
            ...

        Return :
        --------

        x_arr, y_arr : Array of x and y point(s).
        
        """
        x_arr, y_arr = [xi], [yi]
        while (xi + h <= xf):
            if (xi + h ) > xf:
                h = xf - xi
            k1 = dydx(xi, yi)
            k2 = dydx(xi + 1/4 * h, yi + 1/4 * k1 * h)
            k3 = dydx(xi + 1/4 * h, yi + 1/8 * k1 * h + 1/8 * k2 * h)
            k4 = dydx(xi + 1/2 * h, yi - 1/2 * k2 * h + k3 * h)
            k5 = dydx(xi + 3/4 * h, yi + 3/16 * k1 * h + 9/16 * k4 * h)
            k6 = dydx(xi + h , yi - 3/7 * k1 * h + 2/7 * k2 * h + 12/7 * k3 * h - 12/7 * k4 * h + 8/7 * k5 * h)
            yi = yi + 1/90 * (7 * k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6)*h
            xi += h
            x_arr.append(xi)
            y_arr.append(yi)
        return x_arr, y_arr
    
    
    
    

