""" 
Author  : Mehmet Gokcay Kabatas
Mail    : mgokcaykdev@gmail.com
Version : 0.1
Date    : 03/12/2019
Update  : 05/12/2019
Python  : 3.6.5

Update Note : Adding Descriptions.

This script written by @Author for personal usage. 

"""

class RootFind():
    """ Some analytic function's root cannot find analytically. 
    To find root, some numerical approaches used. This class has some 
    methods. 

        @ Methods : 
        - BiSection
        - Newton - Raphson Method
        - Secant 
        - Simpe Fixed-Point Iteration (Just named Iteration)

        @ Note : 
        Before use the methods, you need to set some parameters.
        To do it, you can call `@setParams` method.

        @ Usage :
        ...
        def f(x):
            return pow(x,2)-2
        
        def df(x):
            return 2x

        rf = RootFind()
        rf.setParams(f,df)
        rf.@Methods
     """

    def setParams(self, func=None, der=None, err=1e-13):
        """ To find root, you need to set some properties.
        
        Arguments :
        -------------

        func : Target function which argument depend on 'x'.

            @ Example :
            def f(x):
                return (pow(x,2)-2)
            
            ... 
            .setParams(func=f)
            ...

        der : Target function's derivative function
        which argument also depend on 'x'.

            @ Example :
            def df(x):
                return (2*x)
            
            ... 
            .setParams(func=f, der= df)
            ...

        err : Desired error. Default equal 1e-13.

        """
        self.func = func
        self.der = der
        self.err = err

    def BiSection(self, x0, x1) -> float:
        """ BiSection method.
        
        Arguments :
        -----------
        
        x0 : Initial guess. \n
        x1 : Initial guess.

        Return :
        --------
        x : Root of `@func`.

        """  
        if self.func == None:
            raise Exception("Function (func) should be declared in setParams.")     
        while (self.func(x0) * self.func(x1) < 0 ):
            x2 = x0 + (x1 - x0) / 2
            print(x2)
            if (self.func(x0) * self.func(x2) > 0):
                x0 = x2
            else:
                x1 = x2            
            if abs(self.func(x0) - self.func(x1)) < self.err:
                return x2
        else:
            
            raise Exception('Guess are not proper.\nFunc(x0) * Func(x1) > 0. Select proper guess.')
            return None
        
    def NewtonRaphson(self, x0) -> float:
        """ Newton-Raphson method.

            @Note : Need setted <der> argument in setParams.

        Arguments :
        -----------
        
        x0 : Initial guess. 

        Return :
        --------
        x : Root of `@func`.

        """
        err = 100
        if self.func == None:
            raise Exception("Function (func) should be declared in setParams.")
        if self.der == None:
            raise Exception("Derivative function (der) should be declared in setParams.")
        while ( err > self.err):
            x_n1 = x0 - (self.func(x0) / self.der(x0))
            err = abs(self.func(x0) - self.func(x_n1))
            x0 = x_n1        
        return x_n1

    def Secant(self, x0, x1) -> float:
        """ Secant method.

        Arguments :
        -----------
        
        x0 : Initial guess. \n
        x1 : Initial guess. 

        Return :
        --------
        x : Root of `@func`.

        """
        err = 100
        if self.func == None:
            raise Exception("Function (func) should be declared in setParams.")
        while ( err > self.err):
            x_2 = (x0 * self.func(x1) - x1 * self.func(x0))
            x_2 /= (self.func(x1) - self.func(x0))
            err = abs(self.func(x_2)) 
            x0 = x1
            x1 = x_2       
        return x_2

    def Iteration(self, x0):
        """ Simple Fixed-Iteration method.

        Arguments :
        -----------
        
        x0 : Initial guess. 

        Return :
        --------
        x : Root of `@func`.

        """
        err = 100        
        if self.func == None:
            raise Exception("Function (func) should be declared in setParams.")
        while ( err > self.err):            
            x_1 = self.func(x0)            
            err = abs(x_1 - x0) 
            x0 = x_1            
        return x_1