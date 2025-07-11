#!/usr/bin/env python3
"""
General idea:
- Trying to minimize functions with kinks/measurement error, so I can't just use a simple solver. For example, f(x) = (x-1)**2 + error
- Two cases: single grid (solving for x* only), multigrid (solving for x2*(x1) = min_x2 f(x1, x2))

Potential changes:
- Add Savitzy - Golan alternative to using polynomial method (note that I need an intermediary step where I interpolate the function to ensure I have equal gaps between the points I input into the SG filter)
"""
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

import copy
import datetime
import decimal
import functools
try:
    import matplotlib.pyplot as plt
except Exception:
    None
import multiprocessing
import numpy as np
import shutil
import subprocess
import time

# Imported Functions:{{{1
sys.path.append(str(__projectdir__ / Path('submodules/python-general-func/')))
from print_func import printorwrite
printorwrite = printorwrite

sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
from poly_fit_func import polyestfit_getvec

# Basic Functions:{{{1
def getrangetorun(lb, ub, maxinterval, existingrange = None, decimalplaces = 8):
    """
    maxinterval is the max gap between each element in lb, ub
    existingrange are the points I've already run

    Need to specify decimal places to ensure that the numbers I input are consistently the same.
    """

    if existingrange is None:
        existingrange = []

    vals = np.arange(lb, ub, maxinterval)
    dropi = []
    for i, val in enumerate(vals):
        for existing in existingrange:
            # subtract 1e-9 so that if I do 1.00, 1.01, 1.02 with a maxinterval of 0.01 then I would still include 1.01 if 1.00 and 1.02 already exist
            if abs(float(existing) - val) < maxinterval - 1e-9:
                dropi.append(i)
                break

    vals = [vals[i] for i in range(len(vals)) if i not in dropi]
    # vals = [val for val in vals if val not in existingrange]
    vals = [f"{val:.{decimalplaces}f}" for val in vals]

    return(vals)


def getrangetorun_test():
    print(getrangetorun(0, 2.0001, 1, []))
    print(getrangetorun(0, 2.0001, 1, [1]))
    print(getrangetorun(0, 2.0001, 1, [0.9]))
    print(getrangetorun(0, 2.0001, 1, [0.9, 2.1]))
    print(getrangetorun(0, 2.0001, 1, [0.9, 2.9]))
    print(getrangetorun(0, 2.0001, 1, [-3.5, -2.6, -1.7, -0.8, 0.1, 1.0, 1.9, 2.8]))


def getcurrentstate_singlegrid(savefolder, xvalues = None):
    """
    By default I take all values. xvalues allows me to specify which inputs I consider.
    Returns values as floats
    """
    if xvalues is None:
        xstrs = os.listdir(savefolder)
    else:
        xstrs = xvalues
    inputlist = []
    outputlist = []

    # since xstrs are strings (to preserve decimals, I need to sort them by floats rather than strings)
    xstrdict = {float(xstr): xstr for xstr in xstrs}

    for xstrfloat in sorted(xstrdict):
        xstr = xstrdict[xstrfloat]

        with open(os.path.join(savefolder, xstr)) as f:
            text = f.read()

        inputval = float(text.split(',')[0])
        outputval = float(text.split(',')[1])

        inputlist.append(inputval)
        outputlist.append(outputval)

    return(inputlist, outputlist)


def saveoutput_singlegrid(savefolder, inputvalue, outputvalue, otherinfo = None):
    try:
        os.mkdir(savefolder)
    except Exception:
        None

    if otherinfo is None:
        otherinfo = ''

    savefile = os.path.join(savefolder, str(inputvalue))
    with open(savefile, 'w+') as f:
        f.write(str(inputvalue) + ',' + str(outputvalue) + ',' + str(otherinfo) + '\n')

    return(savefile)


# Processing:{{{1
def finishedprocessingqsub(filenames, timesleep = 300, printdetails = True, outputfilename = None, totaltimemax = None):
    """
    Was told to stop using qsub... so currently not in use
    """
    starttime = datetime.datetime.now()
    while True:
        # can either look at qstat or go through list of files to get how many files to do
        # advantage of qstat method is that if something gets dropped from qstat, I will see this
        output = subprocess.check_output('qstat', shell = True)
        numfilenames_qstat = len(output.decode('utf-8').splitlines())

        # just get how many files completed
        for filename in copy.deepcopy(filenames):
            if os.path.isfile(filename):
                filenames.remove(filename)
        numfilenames = len(filenames)

        # continue only once completed initial iteration
        if numfilenames == 0:
            break

        if printdetails is True:
            printorwrite('Filenames to do: ' + str(filenames) + '. Num filenames to do by counting files: ' + str(numfilenames) + ' Num filenames to do by qstat: ' + str(numfilenames_qstat) + '. Datetime: ' + str(datetime.datetime.now()), outputfilename)

        if (datetime.datetime.now() - starttime).seconds > totaltimemax:
            print('Total time exceeded for this iteration.')
            return(filenames)

        # wait a bit
        time.sleep(timesleep)

    # return empty file list signifying complete
    return([])


# Rangefunclist Functions:{{{1
def getrangetorun_singlegrid(lb, ub, maxinterval, inputvalues, outputvalues, decimalplaces = 8):
    """
    This is similar to the standard getrangetorun function except that it takes savefolder as an argument
    I can then use this as one of the functions in rangefunclist
    """
    rangetorun = getrangetorun(lb, ub, maxinterval, existingrange = inputvalues, decimalplaces = decimalplaces)
    return(rangetorun)


def rangearoundmin(widthoneside, maxinterval, inputvalues, outputvalues):
    # compute the minimum
    argmin = np.argmin(outputvalues)
    themin = inputvalues[argmin]

    lb = themin - widthoneside
    ub = themin + widthoneside

    # get range of values to run
    rangetorun = getrangetorun(lb, ub, maxinterval, inputvalues)

    return(rangetorun)

    
# Solve Single Grid Model (Computing x*):{{{1
def solvesinglegrid(rootsavefolder, singlerunfunc, rangefunclist, printdetails = True, numprocesses = 1, otherparamstopass = None, initialfolder = None):
    """
    Iterate over a grid to find the minimum of a function singlerunfunc
    rootsavefolder is the root folder; I then have a different folder for each stage to ensure I can rerun without issues (otherwise adding new values may affect what rangefunclist does)
    rangefunclist is a list of functions that take the current savefolder and compute the values to try next for the function
    numprocesses allows me to do multiprocessing
    otherparamstopass allows other parameters to be passed to the function
    initialfolder means that I copy the first stage folder from somewhere else before beginning
    """

    if rangefunclist is None:
        rangefunclist = []

    for i in range(len(rangefunclist)):
        if i > 0:
            savefolderstagem1 = savefolderstage
        elif initialfolder is not None:
            if not os.path.isdir(initialfolder):
                raise ValueError('initialfolder is not a folder: ' + str(initialfolder) + '.')
            savefolderstagem1 = initialfolder
        else:
            savefolderstagem1 = None
        savefolderstage = rootsavefolder / Path('stage' + str(i))
        # if current stage exists then just use as is (assuming may have already started running the folder in this case)
        if not os.path.isdir(savefolderstage):
            # otherwise copy or make new folder appropriately
            if savefolderstagem1 is not None:
                # copy across prior stage so can use prior runs in next stage
                # the reason we don't just use the same folder for each run is that after adding the stage1 results, that could change the polynomial that we select after stage0
                shutil.copytree(savefolderstagem1, savefolderstage)
            else:
                os.makedirs(savefolderstage)

        # get input and output values from last iteration
        inputlist, outputlist = getcurrentstate_singlegrid(savefolderstage)

        # updated range to consider
        rangetorun = rangefunclist[i](inputlist, outputlist)

        # remove files which already done
        rangetorun2 = []
        for inputvalue in rangetorun:
            if not os.path.isfile(os.path.join(savefolderstage, inputvalue)):
                rangetorun2.append(inputvalue)
        rangetorun = rangetorun2

        if len(rangetorun) == 0:
            if printdetails is True:
                print('Already completed iteration ' + str(i) + '.')
        else:
            if printdetails is True:
                print('Starting iteration ' + str(i) + '. Number of files: ' + str(len(rangetorun)) + '.')
                
        elements = [(savefolderstage, inputvalue, otherparamstopass) for inputvalue in rangetorun]

        if numprocesses == 1:
            for element in elements:
                singlerunfunc(element)
        else:
            pool = multiprocessing.Pool(numprocesses)
            pool.map(singlerunfunc, elements, chunksize = 1)

        # check ran correctly
        rangetorun2 = []
        for inputvalue in rangetorun:
            if not os.path.isfile(os.path.join(savefolderstage, inputvalue)):
                rangetorun2.append(inputvalue)
        rangetorun = rangetorun2
        if len(rangetorun) > 0:
            raise ValueError('Some filenames failed to run: ' + str(rangetorun) + '.')


# Solving Single Grid Examples:{{{1
def singlegridmodel_ex(element):
    # seeding is important in this multiprocessing
    if False:
        # set seed so always get same result
        np.random.seed(0)
    else:
        # draw a new random seed
        # required with multiprocessing otherwise child processes may have same seed (though does notalways seem to be the case)
        np.random.seed()

    # these are the three arguments that are passed to the single run function
    savefolderstage, inputvalue, otherparams = element

    # set otherparams to be the minimizing value
    minvalue = otherparams
    if minvalue is None:
        minvalue = 0

    # function I'm minimizing
    outputvalue = (float(inputvalue) - minvalue)**2 + np.random.normal() * 0.1
    otherinfo = None

    # I need to save the output in a standard way
    saveoutput_singlegrid(savefolderstage, inputvalue, outputvalue, otherinfo = otherinfo)


def solvesinglegrid_ex_singleprocess_initonly():
    # set seed so always get same result
    # np.random.seed(0)

    # remove
    testfolder = __projectdir__ / Path('tests/singlegrid_singleprocess_initonly/')
    # if os.path.isdir(testfolder):
    #     shutil.rmtree(testfolder)

    if not os.path.isdir(testfolder):
        os.makedirs(testfolder)

    func_stage0 = functools.partial(getrangetorun_singlegrid, -3, 3.001, 0.1)
    rangefunclist = [func_stage0]

    solvesinglegrid(testfolder, singlegridmodel_ex, rangefunclist)

    inputlist, outputlist = getcurrentstate_singlegrid(testfolder / Path('stage0'))
    argmin = np.argmin(outputlist)
    print('Minimizing argument: ' + str(inputlist[argmin]) + '. Value: ' + str(outputlist[argmin]) + '.')


def solvesinglegrid_ex_singleprocess_twostages():
    # set seed so always get same result
    # np.random.seed(0)

    # remove
    testfolder = __projectdir__ / Path('tests/singlegrid_singleprocess_twostages/')
    if os.path.isdir(testfolder):
        shutil.rmtree(testfolder)

    if not os.path.isdir(testfolder):
        os.makedirs(testfolder)

    func_stage0 = functools.partial(getrangetorun_singlegrid, -3, 3.001, 0.1)
    func_stage1 = functools.partial(rangearoundmin, 0.5, 0.02)
    rangefunclist = [func_stage0, func_stage1]

    solvesinglegrid(testfolder, singlegridmodel_ex, rangefunclist)

    inputlist, outputlist = getcurrentstate_singlegrid(testfolder / Path('stage1'))
    argmin = np.argmin(outputlist)
    print('Minimizing argument: ' + str(inputlist[argmin]) + '. Value: ' + str(outputlist[argmin]) + '.')


def solvesinglegrid_ex_multiprocess_twostages():
    """
    If I want to seed, I should do so in the single run function
    """

    # remove
    testfolder = __projectdir__ / Path('tests/singlegrid_multiprocess_twostages/')
    if os.path.isdir(testfolder):
        shutil.rmtree(testfolder)

    if not os.path.isdir(testfolder):
        os.makedirs(testfolder)

    func_stage0 = functools.partial(getrangetorun_singlegrid, -3, 3.001, 0.1)
    func_stage1 = functools.partial(rangearoundmin, 0.5, 0.02)
    rangefunclist = [func_stage0, func_stage1]

    # add otherparams here
    # in the function I set otherparams equal to the minimizing value
    # this is a way of allowing me to change parameters in the function without having to specify a different function to run each time
    otherparamstopass = None

    solvesinglegrid(testfolder, singlegridmodel_ex, rangefunclist = rangefunclist, numprocesses = multiprocessing.cpu_count(), otherparamstopass = otherparamstopass)

    inputlist, outputlist = getcurrentstate_singlegrid(testfolder / Path('stage1'))
    argmin = np.argmin(outputlist)
    print('Minimizing argument: ' + str(inputlist[argmin]) + '. Value: ' + str(outputlist[argmin]) + '.')


def solvesinglegrid_ex_multiprocess_copyfirst():
    """
    If I want to seed, I should do so in the single run function
    """

    # remove
    testfolder = __projectdir__ / Path('tests/singlegrid_multiprocess_copyfirst/')
    if os.path.isdir(testfolder):
        shutil.rmtree(testfolder)

    if not os.path.isdir(__projectdir__ / Path('tests/singlegrid_multiprocess_twostages/')):
        raise ValueError('Need to run solvesinglegrid_ex_multiprocess_twostages first.')

    if not os.path.isdir(testfolder):
        os.makedirs(testfolder)

    # copying stage0 from another folder
    func_stage1 = functools.partial(rangearoundmin, 0.5, 0.02)
    rangefunclist = [func_stage1]

    # add otherparams here
    # in the function I set otherparams equal to the minimizing value
    # this is a way of allowing me to change parameters in the function without having to specify a different function to run each time
    otherparamstopass = None

    solvesinglegrid(testfolder, singlegridmodel_ex, rangefunclist = rangefunclist, numprocesses = multiprocessing.cpu_count(), otherparamstopass = otherparamstopass, initialfolder = __projectdir__ / Path('tests/singlegrid_multiprocess_twostages/stage0/'))

    inputlist, outputlist = getcurrentstate_singlegrid(testfolder / Path('stage0'))
    argmin = np.argmin(outputlist)
    print('Minimizing argument: ' + str(inputlist[argmin]) + '. Value: ' + str(outputlist[argmin]) + '.')


# Multi Grid General Functions:{{{1
def getcurrentstate_multigrid(savefolder, x1valdict = None):
    """
    Get the currentstate for a multigrid savefolder
    (x1, x2) are the points I'm considering where I want to find x1^star(x2)
    Then the structure of the saves are savefolder/x1/x2

    x1 is a string while x2 is a float
    """
    x1values = os.listdir(savefolder)

    if x1valdict is None:
        x1values_sortdict = {float(x1value): x1value for x1value in x1values}
    else:
        x1values_sortdict = {float(x1value): x1value for x1value in list(x1valdict)}

    inputdict = {}
    # call by sorted x1values as floats since that yields the correct order
    for x1valuefloat in sorted(x1values_sortdict):
        x1value = x1values_sortdict[x1valuefloat]
        if x1valdict is None:
            x2values = None
        else:
            x2values = x1valdict[x1value]

        ret = getcurrentstate_singlegrid(os.path.join(savefolder, x1value), xvalues = x2values)
        # ret is both x2values and welfare
        # drop if len of x2values is 0
        if len(ret[0]) > 0:
            inputdict[x1value] = ret

    return(inputdict)


def saveoutput_multigrid(savefolder, x1value, x2value, outputvalue, otherinfo = None):
    savefolder2 = os.path.join(savefolder, x1value)
    try:
        os.makedirs(savefolder2)
    except Exception:
        None
    saveoutput_singlegrid(savefolder2, x2value, outputvalue, otherinfo = otherinfo)


# How Update Multigrid Range:{{{1
def getrangetorun_multigrid(x1values, lb, ub, maxinterval, inputdict, decimalplaces = 8, addlbfunc = None, addubfunc = None):
    """
    Get the initial range of values on which to consider a grid search. If the currentvalues already exist then use those.

    Note that we don't actually need outputvalues.

    addlbfunc(x1value) defines a second lower bound for a given x1value. If this is higher than lb, this applies, otherwise lb applies
    """

    # I think I should do this outside of the iterative function to ensure I always get the same
    # convert to string to avoid issues with saving floats
    # x1values = fixfloats(x1values, decimalplaces = decimalplaces)
    outputdict = {}
    for x1value in x1values:
        if x1value in inputdict:
            x2values = inputdict[x1value][0]
        else:
            x2values = []

        if addlbfunc is not None:
            thislb = np.max([lb, addlbfunc(x1value)])
        else:
            thislb = lb
        if addubfunc is not None:
            thisub = np.min([ub, addubfunc(x1value)])
        else:
            thisub = ub

        # get range of values to run
        outputdict[x1value] = getrangetorun(thislb, thisub, maxinterval, x2values)

    return(outputdict)


def rangearoundfittedcurve(x1values_new, widthoneside, maxinterval, fitfunc, inputdict, addlbfunc = None, addubfunc = None, alternativesavefolder = None):
    """
    fitfunc is how I compute estimated values for what new x2values will be given new x1values
    i.e. x2values_new_fit = fitfunc(x1values_current, x2values_current, x1values_new)
    """

    if alternativesavefolder is not None:
        # want possibility of getting fitted curve from a folder other than the one I'm considering
        # for example, let's say I want to minimize f_a, f_b over similar values
        # if f_a and f_b have similar minima, to consider x2^star(x1) for f_b for x1=[1, 2], I can compute the x2^star(x1) implied by f_a for [1, 2]
        inputdict_current = getcurrentstate_multigrid(alternativesavefolder)
    else:
        inputdict_current = inputdict

    # compute the minimum
    x1values = []
    x2values = []
    for x1value in inputdict_current:

        outputvalues = inputdict_current[x1value][1]
        inputvalues = inputdict_current[x1value][0]

        # if smoothing is not None, smooth outputvalues to get better estimate of current x2^star(x1)
        # smoothing should be an integer
        # if smoothing not in [None, False]:
        #     # don't want smoothing to be too high since that would mean they'd be very few x2 points to choose from
        #     smoothing = np.min([smoothing, np.max([len(outputvalues) // 4, 1])])
        #
        #     # do smoothing
        # sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
        # from smooth_func import masmoothseries
        #     outputvalues = masmoothseries(outputvalues, smoothing)
        #
        #     # remove nan values
        #     na_val = ~np.isnan(outputvalues)
        #     inputvalues = np.array(inputvalues)[na_val]
        #     outputvalues = outputvalues[na_val]
        
        argmin = np.argmin(outputvalues)
        x2value = inputvalues[argmin]

        x1values.append(float(x1value))
        x2values.append(x2value)

    # # compute polynomial fit
    # sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
    # from poly_fit_func import polyestimate
    # betahat = polyestimate(x1values, coeff, x2values)
    #
    # if printdetails is True:
    #     printorwrite('Estimated polynomial: ' + ' + '.join([str(betahat[i]) + 'x**' + str(i) for i in range(len(betahat))]) + '.', outputfilename)
    #
    # # estimated x2values_new based upon x1values_new I'm going to run
    # sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
    # from poly_fit_func import polyfit
    # x2valueshat = polyfit(coeff, betahat, [float(x1value) for x1value in x1values_new])
    # x2valueshat = x2valueshat.reshape([len(x2valueshat)])


    x2valueshat = fitfunc(x1values, x2values, [float(x1value) for x1value in x1values_new])


    outputdict = {}
    for i in range(len(x1values_new)):
        x1value = x1values_new[i]
        x2valuehat = x2valueshat[i]

        lb = x2valuehat - widthoneside
        ub = x2valuehat + widthoneside

        # additional functions to give second lb, ub
        # ensures that lb does not fall below a value for which the simulation would fail
        if addlbfunc is not None:
            lb = np.max([lb, addlbfunc(x1value)])
        if addubfunc is not None:
            ub = np.min([ub, addubfunc(x1value)])


        # get existing x2values for each x1value
        if x1value in inputdict:
            inputvalues = inputdict[x1value][0]
        else:
            inputvalues = []

        # get range of values to run
        rangetorun = getrangetorun(lb, ub, maxinterval, inputvalues)

        outputdict[x1value] = rangetorun

    return(outputdict)

    
def rangearoundfittedcurve_poly(x1values_new, widthoneside, maxinterval, coeff, inputdict, ismaximum = False, printdetails = True, addlbfunc = None, addubfunc = None, alternativesavefolder = None):
    """
    fitfunc is how I compute estimated values for what new x2values will be given new x1values
    i.e. x2values_new_fit = fitfunc(x1values_current, x2values_current, x1values_new)
    """

    # set fitfunc to be a polynomial
    fitfunc = functools.partial(polyestfit_getvec, coeff, printdetails = printdetails)

    outputdict = rangearoundfittedcurve(x1values_new, widthoneside, maxinterval, fitfunc, inputdict, addlbfunc = addlbfunc, addubfunc = addubfunc, alternativesavefolder = alternativesavefolder)

    return(outputdict)
    

# Solve Multi Grid Model:{{{1
def solvemultigrid(rootsavefolder, singlerunfunc, rangefunclist, printdetails = True, numprocesses = 1, otherparamstopass = None, initialfolder = None):
    """
    Key difference between this and solvesinglegrid: f1 is now a function of two values f(x1, x2)
    We want to find x1^star given a value of x2
    
    Therefore, the input to rangefunclist[i] is now a dictionary for each value of x2 that I have run values for which yields inputdict[x2] = inputvalues, outputvalues where inputvalues are the x1 values I ran and outputvalues are the corresponding f(x1, x2) values
    And the output of rangefunclist[i] is a dictionary 
    """

    for i in range(len(rangefunclist)):
        if i > 0:
            savefolderstagem1 = savefolderstage
        elif initialfolder is not None:
            if not os.path.isdir(initialfolder):
                raise ValueError('initialfolder is not a folder: ' + str(initialfolder) + '.')
            savefolderstagem1 = initialfolder
        else:
            savefolderstagem1 = None
        savefolderstage = rootsavefolder / Path('stage' + str(i))
        if not os.path.isdir(savefolderstage):
            if savefolderstagem1 is not None:
                # copy across prior stage so can use prior runs in next stage
                # the reason we don't just use the same folder for each run is that after adding the stage1 results, that could change the polynomial that we select after stage0
                if numprocesses > 1:
                    try:
                        shutil.copytree(savefolderstagem1, savefolderstage)
                    except Exception:
                        None
                else:
                    shutil.copytree(savefolderstagem1, savefolderstage)
            else:
                if numprocesses > 1:
                    try:
                        os.makedirs(savefolderstage)
                    except Exception:
                        None
                else:
                    os.makedirs(savefolderstage)

        # get input and output values from last iteration
        inputdict = getcurrentstate_multigrid(savefolderstage)

        # updated range to consider
        outputdict = rangefunclist[i](inputdict)

        # remove x1 folders that we are no longer considering
        # why? because these may have limited x2 values so it would be better to consider the fit using only the x1 values for the latest iteration
        # this must happen after I compute outputdict (since rangefunclist[i] computes new list based on old values)
        for x1valueold in os.listdir(savefolderstage):
            if x1valueold not in outputdict:
                shutil.rmtree(savefolderstage / x1valueold)

        # remove elements from outputdict which already done
        outputdict2 = {}
        for x1value in outputdict:
            for x2value in outputdict[x1value]:
                if not os.path.isfile(os.path.join(savefolderstage, x1value, x2value)):
                    if not x1value in outputdict2:
                        outputdict2[x1value] = []
                    outputdict2[x1value].append(x2value)
        outputdict = outputdict2

        if len(outputdict) == 0:
            if printdetails is True:
                print('Already completed stage ' + str(i) + '.')
        else:
            if printdetails is True:
                print('Starting stage ' + str(i) + '. Number of files: ' + str(sum(map(len, outputdict.values()))) + '.')
                
        elements = [(savefolderstage, x1value, x2value, otherparamstopass) for x1value in outputdict for x2value in outputdict[x1value]]

        if numprocesses == 1:
            for element in elements:
                singlerunfunc(element)
        else:
            pool = multiprocessing.Pool(numprocesses)
            pool.map(singlerunfunc, elements, chunksize = 1)

        # check all filenames ran
        outputdict2 = {}
        for x1value in outputdict:
            for x2value in outputdict[x1value]:
                if not os.path.isfile(os.path.join(savefolderstage, x1value, x2value)):
                    if not x1value in outputdict2:
                        outputdict2[x1value] = []
                    outputdict2[x1value].append(x2value)
        if len(outputdict2) > 0:
            raise ValueError('Some filenames failed to run: ' + str(outputdict2) + '.')


# Analysis Multi Grid Model:{{{1
def getmin_multigrid(savefolder):
    if not os.path.isdir(savefolder):
        raise ValueError('Not a folder: ' + str(savefolder) + '.')

    retdict = getcurrentstate_multigrid(savefolder)

    x1values = []
    x2values = []
    for x1value in retdict:
        inputvalues = retdict[x1value][0]
        outputvalues = retdict[x1value][1]
        
        argmin = np.argmin(outputvalues)
        x2value = inputvalues[argmin]

        x1values.append(float(x1value))
        x2values.append(float(x2value))

    return(x1values, x2values)


def getfittedvals_multigrid(fitfunc, x1values, x2values, x1values_tofit):
    """
    This just uses fitfunc to estimate the appropriate fit of x2values on x1values and then estimates x2values_tofit based upon x1values_tofit
    """
    return(fitfunc(x1values, x2values, x1values_tofit))


def getdetails_multigrid(savefolder, fitfunc = None, pltshow = False, pltsavename = None, xlabel = None, ylabel = None):
    if not os.path.isdir(savefolder):
        raise ValueError('Not a folder: ' + str(savefolder) + '.')
    x1values, x2values = getmin_multigrid(savefolder)

    plt.plot(x1values, x2values, 'o', label = 'Simulation')

    if fitfunc is not None:
        x2values_fitted = fitfunc(x1values, x2values, x1values)
        plt.plot(x1values, x2values_fitted, label = 'Best Fit')

        plt.legend()

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if pltsavename is not None:
        plt.savefig(pltsavename)
    if pltshow is True:
        plt.show()
    plt.close()

    if fitfunc is not None:
        fitfunc2 = functools.partial(fitfunc, x1values, x2values)
        return(fitfunc2)


# Solving Multi Grid Examples:{{{1
def multigridmodel_ex(element):
    # seeding is important in this multiprocessing
    if False:
        # set seed so always get same result
        np.random.seed(0)
    else:
        # draw a new random seed
        # required with multiprocessing otherwise child processes may have same seed (though does notalways seem to be the case)
        np.random.seed()

    # these are the four arguments that are passed to the single run function
    savefolderstage, x1value, x2value, otherparams = element

    # set otherparams to be the minimizing value
    minvalue = otherparams
    if minvalue is None:
        minvalue = 0

    # function I'm minimizing
    outputvalue = (float(x1value) + float(x2value) - minvalue)**2 + np.random.normal() * 0.1
    otherinfo = None

    # I need to save the output in a standard way
    saveoutput_multigrid(savefolderstage, x1value, x2value, outputvalue, otherinfo = otherinfo)


def solvemultigrid_ex_singleprocess_initonly(pltshow = False):
    # set seed so always get same result
    # np.random.seed(0)

    # remove
    testfolder = __projectdir__ / Path('tests/multigrid_singleprocess_initonly/')
    if os.path.isdir(testfolder):
        shutil.rmtree(testfolder)

    if not os.path.isdir(testfolder):
        os.makedirs(testfolder)

    x1values_stage0 = getrangetorun(-3, 3.001, 0.5)
    func_stage0 = functools.partial(getrangetorun_multigrid, x1values_stage0, -3, 3.001, 0.5)
    rangefunclist = [func_stage0]

    solvemultigrid(testfolder, multigridmodel_ex, rangefunclist)

    fitfunc = functools.partial(polyestfit_getvec, 2, printdetails = True)
    getdetails_multigrid(testfolder / Path('stage0'), fitfunc = fitfunc, pltshow = pltshow)


def solvemultigrid_ex_singleprocess_twostages(pltshow = False):
    # set seed so always get same result
    # np.random.seed(0)

    # remove
    testfolder = __projectdir__ / Path('tests/multigrid_singleprocess_twostages/')
    if os.path.isdir(testfolder):
        shutil.rmtree(testfolder)

    if not os.path.isdir(testfolder):
        os.makedirs(testfolder)

    x1values_stage0 = getrangetorun(-3, 3.001, 0.5)
    x1values_stage1 = getrangetorun(-3, 3.001, 0.5)
    func_stage0 = functools.partial(getrangetorun_multigrid, x1values_stage0, -3, 3.001, 0.5)
    # this gets a line of best fit for \\hat{x2}^\\star(x1) using the previous iteration values
    # this line of best fit is a polynomial of order 3
    # then run values within 0.5 of the \\hat{x2}^\\star(x1) value for each x1 in x1values_stage1
    # run each value separated by 0.1 that is not already approximately in the data
    func_stage1 = functools.partial(rangearoundfittedcurve_poly, x1values_stage1, 0.5, 0.1, 3)
    rangefunclist = [func_stage0, func_stage1]

    solvemultigrid(testfolder, multigridmodel_ex, rangefunclist)

    fitfunc = functools.partial(polyestfit_getvec, 2, printdetails = True)
    getdetails_multigrid(testfolder / Path('stage1'), fitfunc = fitfunc, pltshow = pltshow)


def solvemultigrid_ex_multiprocess_twostages(pltshow = False):
    # set seed so always get same result
    # np.random.seed(0)

    # remove
    testfolder = __projectdir__ / Path('tests/multigrid_multiprocess_twostages/')
    if os.path.isdir(testfolder):
        shutil.rmtree(testfolder)

    if not os.path.isdir(testfolder):
        os.makedirs(testfolder)

    x1values_stage0 = getrangetorun(-3, 3.001, 0.5)
    x1values_stage1 = getrangetorun(-3, 3.001, 0.5)
    func_stage0 = functools.partial(getrangetorun_multigrid, x1values_stage0, -3, 3.001, 0.5)
    # this gets a line of best fit for \\hat{x2}^\\star(x1) using the previous iteration values
    # this line of best fit is a polynomial of order 3
    # then run values within 0.5 of the \\hat{x2}^\\star(x1) value for each x1 in x1values_stage1
    # run each value separated by 0.1 that is not already approximately in the data
    func_stage1 = functools.partial(rangearoundfittedcurve_poly, x1values_stage1, 0.5, 0.1, 3)
    rangefunclist = [func_stage0, func_stage1]

    solvemultigrid(testfolder, multigridmodel_ex, rangefunclist, numprocesses = multiprocessing.cpu_count())

    fitfunc = functools.partial(polyestfit_getvec, 2, printdetails = True)
    getdetails_multigrid(testfolder / Path('stage1'), fitfunc = fitfunc, pltshow = pltshow)


def solvemultigrid_ex_multiprocess_copyfirst(pltshow = False):
    # set seed so always get same result
    # np.random.seed(0)

    # remove
    testfolder = __projectdir__ / Path('tests/multigrid_multiprocess_copyfirst/')
    if os.path.isdir(testfolder):
        shutil.rmtree(testfolder)

    if not os.path.isdir(testfolder):
        os.makedirs(testfolder)

    x1values_stage0 = getrangetorun(-3, 3.001, 0.5)
    x1values_stage1 = getrangetorun(-3, 3.001, 0.5)
    # this gets a line of best fit for \\hat{x2}^\\star(x1) using the previous iteration values
    # this line of best fit is a polynomial of order 3
    # then run values within 0.5 of the \\hat{x2}^\\star(x1) value for each x1 in x1values_stage1
    # run each value separated by 0.1 that is not already approximately in the data
    func_stage1 = functools.partial(rangearoundfittedcurve_poly, x1values_stage1, 0.5, 0.1, 3)
    rangefunclist = [func_stage1]

    solvemultigrid(testfolder, multigridmodel_ex, rangefunclist, numprocesses = multiprocessing.cpu_count(), initialfolder = __projectdir__ / Path('tests/multigrid_singleprocess_twostages/stage0/'))

    fitfunc = functools.partial(polyestfit_getvec, 2, printdetails = True)
    getdetails_multigrid(testfolder / Path('stage0'), fitfunc = fitfunc, pltshow = pltshow)


# Test All:{{{1
def testall():
    solvesinglegrid_ex_singleprocess_initonly()
    solvesinglegrid_ex_singleprocess_twostages()
    solvesinglegrid_ex_multiprocess_twostages()
    solvesinglegrid_ex_multiprocess_copyfirst()

    solvemultigrid_ex_singleprocess_initonly()
    solvemultigrid_ex_singleprocess_twostages()
    solvemultigrid_ex_multiprocess_twostages()
    solvemultigrid_ex_multiprocess_copyfirst()


# Run:{{{1
if __name__ == "__main__":
    testall()
