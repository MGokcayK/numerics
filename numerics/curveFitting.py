""" 
Author  : Mehmet Gokcay Kabatas
Mail    : mgokcaykdev@gmail.com
Version : 0.1
Date    : 11/12/2019
Update  : 28/05/2020
Python  : 3.6.5

Update Note : Adding exponential regression.

This script written by @Author for personal usage. 

Prerequest : numpy

"""

from numerics.tool import *
import numpy as np

class LinearRegression():
    """
    This class written for numerical methods for Linear Regression application. 

    Arguments :
        -------------
        xValues = x values of curve fitting data.

        yValues = y values of curve fitting data.

    
    \n Class has 2 methods. \n
        - results() : Return of coefficient.
        - standartErrorEstimate() : Return of error estimate.

        @Usage :
        ...
        cf = LinearRegression(xValues, yValues)
        a1, a0 = cf.results()
        error = cf.standartErrorEstimate()
        ...

    """
    def __init__(self, xValues, yValues):
        self._xValues = xValues
        self._yValues = yValues
        self._n = len(xValues)
        self.__compute()

    def __compute(self):
        self._sumX = 0 
        self._sumY = 0
        self._sumXY = 0
        self._sumXSq = 0
        self._sumYSq = 0
        for itemX, itemY in zip(self._xValues, self._yValues):
            self._sumX += itemX
            self._sumY += itemY
            self._sumXY += itemX * itemY
            self._sumXSq += itemX ** 2
            self._sumYSq += itemY ** 2

    def results(self):
        """
        This function return coefficient of Linear Regression.

        Return :
        --------
                Return will be two coefficients which are a1 and a0.
                f(x) = a0 + a1x

            @Usage :
            ...
            cf = LineraRegression(xValues, yValues)
            a1, a0 = cf.results()
            ...
        """
        a1 = self._n * self._sumXY - self._sumX * self._sumY
        a1 /= self._n * self._sumXSq - self._sumX **2
        a0 = arithmaticMean(self._yValues) - a1 * arithmaticMean(self._xValues)        
        return a1, a0

    def standartErrorEstimate(self):
        """
        This function return error estimate of Linear Regression.

        Return :
        --------
                Return will be value of error of Linear Regression.

            @Usage :
            ...
            cf = LineraRegression(xValues, yValues)
            error  = cf.standartErrorEstimate()
            ...
        """
        r = self._n * self._sumXY - self._sumX * self._sumY
        div = (self._n * self._sumXSq - self._sumX ** 2) ** (0.5) * (self._n * self._sumYSq - self._sumY ** 2) ** (0.5)
        r /= div
        return r



class MultipleLinearRegression():
    """
    This class written for numerical methods for Multiple Linear Regression application. 
    \n y = a0 + a1x1 + a2x2 + ...

    Arguments :
    ----------
        xValues = x values of curve fitting data.

        yValues = y values of curve fitting data.

        @Note : x values should be array. \n 
        Ex : 
            x1 = [0,2,2.5, 1, 4, 7]
            x2 = [0, 1, 2, 3, 6, 2]
            cf = MultipleLinearRegression([x1,x2],y)

    
    \n Class has 1 methods. \n
        - results() : Return of coefficient.

        @Usage :
        ...
        cf = MultipleLineraRegression(xValues, yValues)
        coeff = cf.results()
        ...

    """
    def __init__(self, xValues, yValues):
        self.res = PolynomialRegression(xValues,yValues,2)

    def results(self):
        """
        This function return coefficient of Multiple Linear Regression.

        Return :
        --------
                Return will be array of coefficients.\n
                coeff = [a0, a1, a2 ...]
        

            @Usage :
            ...
            cf = LineraRegression(xValues, yValues)
            coeff = cf.results()
            ...
        """
        return self.res.results()



class PolynomialRegression():
    """
    This class written for numerical methods for Polynomial Regression application. 


    Arguments :
        -------------
        xValues = x values of curve fitting data.

        yValues = y values of curve fitting data.

        order = Order of polynomial regression approach.

    \n Class has 1 methods. \n
        - results() : Return of coefficient.

        @Usage :
        ...
        cf = PolynomialRegression(xValues, yValues, order)
        coeff = cf.results()
        ...

    """
    def __init__(self, xValues, yValues, order):
        self._xValues = np.array(xValues)
        self._yValues = np.array(yValues)
        self._n = len(self._yValues)
        self._order = order
        if self._n < order + 1:
            print("Regression cannot process, check order or number of data not enough.")
        else:
            self.__compute()

    def __compute(self):
        self._A = np.empty((self._order + 1,self._order + 1))
        self._B = np.empty((self._order + 1))
        for i in range(self._order + 1):
            for j in range(i+1):
                k = i + j 
                sum = 0
                for l in range(self._n):
                    if (self._xValues.ndim > 1):
                        if j > 0:
                            sum += self._xValues[i-1][l] * self._xValues[j-1][l]
                        else:
                            sum += self._xValues[i-1][l]
                    else:
                        sum += self._xValues[l] ** k
                self._A[i][j] = sum
                self._A[j][i] = sum
                if (i == 0 and j == 0):
                    self._A[i][j] = self._n
            sumB = 0
            for l in range(self._n):
                if (self._xValues.ndim > 1):
                    if i == 0:
                        sumB += self._yValues[l]
                    else:
                        sumB += self._xValues[i-1][l] * self._yValues[l]
                else:
                    sumB += self._xValues[l] ** i * self._yValues[l]
            self._B[i] = sumB
        self._a = np.matmul(self._B, np.linalg.inv(self._A))
    
    def results(self):
        """
        This function return coefficient of Polynomial Regression.

        Return :
        --------
                Return will be array of coefficients.
        

            @Usage :
            ...
            cf = LineraRegression(xValues, yValues)
            coeff = cf.results()
            ...
        """
        return self._a



class Interpolation():
    """
    This class written for numerical methods interpolation.

        @Methods :
        - Linear Interpolation
        - Newton Polynomial Interpolation
        - Linear Splines

        @Usage :
        ...
        interpolate = Interpolation()
        interpolate.@Methods(arguments..)
        ...
    """


    def LinearInterpolation(self, pt1, pt2, value, func):
        """
            This function calculate value of Linear Interpolation w.r.t func.

            Arguments :
            -------------
                pt1 = first point of interpolation.

                pt2 = second point of interpolation.

                value = target value of function. f(value)= ?

                func = target function.

            Return :
            --------
                Value of interpolated function.        

                @Usage :
                ...
                cf = Interpolation()
                y = cf.LinearInterpolation(pt1,pt2,value,func)   
                ...
        """
        res = func(pt1) + (func(pt2) - func(pt1)) 
        res /= (pt2 - pt1) * (value - pt1)
        return res

    def NewtonPolynomialInterpolation(self, x, y, value, order):
        """
            This function calculate value of Newton Polynomial Interpolation.

            Arguments :
            -------------
                x = x values of data.

                y = y values of data.

                valeu = targer value of function. f(value)=?

                order = order of interpolation.

            Return :
            --------
                Value of interpolated function and estimated error.        

                @Usage :
                ...
                cf = Interpolation()
                y, err = cf.NewtonPolynomialInterpolation(x, y, value, order)
                ...
        """
        fdd = np.empty((order,order))
        for i in range(order):
            fdd[i][0] = y[i]
        for j in range(1,order):
            for i in range(order-j):
                fdd[i][j] = (fdd[i+1][j-1] - fdd[i][j-1]) / (x[i+j]-x[i])
        xterm = 1
        yint = np.array([])
        ea = np.empty((order,1))
        yint = np.append(yint,fdd[0][0])
        for ord in range(1,order):
            xterm = xterm * (value - x[ord-1])
            yint2 = yint[ord-1] + fdd[0][ord] * xterm
            ea[ord-1] = yint2 - yint[ord-1]
            yint= np.append(yint,yint2)
        return yint[-1], ea[-1]

    def LinearSplines(self, x, y, value):
        """
            This function calculate value of Linear Spline Interpolation approach.

            Arguments :
            -------------
                x = x values of data.

                y = y values of data.

                valeu = targer value of function. f(value)=?


            Return :
            --------
                Value of interpolated function.     

                @Usage :
                ...
                cf = Interpolation()
                y, err = cf.LinearSplines(x, y, value)
                ...
        """
        for item in reversed(x):
            if item < value:
                ind  = x.index(item)
                m = (y[ind + 1] - y[ind]) / (x[ind + 1] - x[ind])
                res = y[ind] + m * (value - x[ind])
                return res


class ExponentialRegression():
    """
    This class written for numerical methods for Exponential Regression application. 

    Arguments :
        -------------
        xValues = x values of curve fitting data.

        yValues = y values of curve fitting data.

        mode = Type of exponential.

            exp = A*exp(Bx)

            nexp = A*B^(x)

    
    \n Class has 1 methods. \n
        - results() : Return of coefficient.

        @Usage :
        ...
        cf = ExponentialRegression(xValues, yValues, mode)
        A, B = cf.results()
        ...

    """
    def __init__(self, xValues, yValues, mode='exp'):
        self._xValues = xValues
        self._yValues = np.log(yValues)
        self._mode = mode.lower()
        self._n = len(xValues)
        self.__compute()

    def __compute(self):
        self._sumX = 0 
        self._sumY = 0
        self._sumXY = 0
        self._sumXSq = 0
        self._sumYSq = 0
        for itemX, itemY in zip(self._xValues, self._yValues):
            self._sumX += itemX
            self._sumY += itemY
            self._sumXY += itemX * itemY
            self._sumXSq += itemX ** 2
            self._sumYSq += itemY ** 2

    def results(self):
        """
        This function return coefficient of Exponential Regression.

        Return :
        --------
                Return will be two coefficients which are A and B.
    
                for mode 'exp' :
                    
                    f(x) = A * exp(Bx)

                for mode 'nexp' :

                    f(x) = A * B^(x)

            @Usage :
            ...
            cf = ExponentialRegression(xValues, yValues, 'exp')
            A, B = cf.results()
            ...
        """
        a1 = self._n * self._sumXY - self._sumX * self._sumY
        a1 /= self._n * self._sumXSq - self._sumX **2
        a0 = arithmaticMean(self._yValues) - a1 * arithmaticMean(self._xValues)        

        A = np.exp(a0)

        if self._mode == 'exp':
            B = a1
        else:
            B = np.exp(a1)

        return A, B