""" 
Author  : Mehmet Gokcay Kabatas
Mail    : mgokcaykdev@gmail.com
Version : 0.1
Date    : 11/12/2019
Update  : 14/12/2019
Python  : 3.6.5

Update Note : Adding descriptions.

This script written by @Author for personal usage. 

"""
def arithmaticMean(values):
    sum = 0
    for item in values:
        sum += item
    return sum/len(values)

def variance(values):
    sumYi = 0
    sumYiSq = 0
    for item in values:
        sumYiSq += item**2
        sumYi += item
    var = sumYiSq - sumYi**2 / len(values)
    var /= (len(values) - 1)
    return var

def standartDeviation(values):
    return variance(values)**(0.5)

def coeffOfVariation(values):
    sy = standartDeviation(values)
    yBar = arithmaticMean(values)
    return sy/yBar * 100

