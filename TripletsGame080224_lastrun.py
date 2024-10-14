#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.4),
    on October 14, 2024, at 15:33
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware, iohub
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.4'
expName = 'exp'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1280, 1024]
_loggingLevel = logging.getLevel('warning')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Display\\Desktop\\builderTripletTask\\TripletsGame080224_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('el_key_resp') is None:
        # initialise el_key_resp
        el_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='el_key_resp',
        )
    if deviceManager.getDevice('start') is None:
        # initialise start
        start = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='start',
        )
    # create speaker 'sound_pop'
    deviceManager.addDevice(
        deviceName='sound_pop',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=8.0
    )
    # create speaker 'sound_pop2'
    deviceManager.addDevice(
        deviceName='sound_pop2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=8.0
    )
    # create speaker 'sound_pop3'
    deviceManager.addDevice(
        deviceName='sound_pop3',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=8.0
    )
    # create speaker 'sound_pop4'
    deviceManager.addDevice(
        deviceName='sound_pop4',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'sound_pop5'
    deviceManager.addDevice(
        deviceName='sound_pop5',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'sound_pop6'
    deviceManager.addDevice(
        deviceName='sound_pop6',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('start2') is None:
        # initialise start2
        start2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='start2',
        )
    if deviceManager.getDevice('exp_endbreak') is None:
        # initialise exp_endbreak
        exp_endbreak = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='exp_endbreak',
        )
    # create speaker 'sound_pop7'
    deviceManager.addDevice(
        deviceName='sound_pop7',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'sound_pop_2'
    deviceManager.addDevice(
        deviceName='sound_pop_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'sound_pop2_2'
    deviceManager.addDevice(
        deviceName='sound_pop2_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'sound_pop3_2'
    deviceManager.addDevice(
        deviceName='sound_pop3_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'sound_pop4_2'
    deviceManager.addDevice(
        deviceName='sound_pop4_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'sound_pop5_2'
    deviceManager.addDevice(
        deviceName='sound_pop5_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'sound_pop6_2'
    deviceManager.addDevice(
        deviceName='sound_pop6_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'sound_pop7_2'
    deviceManager.addDevice(
        deviceName='sound_pop7_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "eyelinkSetup" ---
    el_key_resp = keyboard.Keyboard(deviceName='el_key_resp')
    
    # Unknown component ignored: Initialize
    
    
    # Unknown component ignored: CameraSetup
    
    
    # --- Initialize components for Routine "initCode" ---
    # Run 'Begin Experiment' code from randomizeImages
    import random, csv, codecs
    #set random seed
    random.seed()
    
    #define rest of the variables
    retProb = random.random()
    rightPos = None
    midPos = None
    leftPos = None
    firstPos = None
    secondPos = None
    thirdPos = None
    TestQuestion = None
    MouseResp = None
    trialType = None
    methodAns = None
    nextTrial = None
    methodCheck = None
    retrievalImage = None
    response = None
    corrAns = None
    decResponse = None
    encodingType = None
    trialPics = []
    retrievalOld = []
    retrievalNew = []
    trial_category = []
    eTrials = None
    retTrials = None
    
    # --- Initialize components for Routine "trial" ---
    Instructions = visual.TextStim(win=win, name='Instructions',
        text='In this task you will review sets of objects in triplets. Your task is to pay attention to where and when these objects appear. Press the spacebar to begin.',
        font='Arial',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    start = keyboard.Keyboard(deviceName='start')
    
    # --- Initialize components for Routine "startRecordingEncoding1" ---
    
    # Unknown component ignored: HostDrawing_Encoding1
    
    
    # Unknown component ignored: StartRecord_Endcoding1
    
    
    # --- Initialize components for Routine "fixation" ---
    fix = visual.ImageStim(
        win=win,
        name='fix', 
        image='images/fixation.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.07, 0.05),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # Unknown component ignored: MarkEvents_fixation
    
    
    # --- Initialize components for Routine "first_view" ---
    Triplet1 = visual.ImageStim(
        win=win,
        name='Triplet1', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-1.0)
    Triplet2 = visual.ImageStim(
        win=win,
        name='Triplet2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-2.0)
    Triplet3 = visual.ImageStim(
        win=win,
        name='Triplet3', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-3.0)
    sound_pop = sound.Sound(
        'A', 
        secs=1.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_pop',    name='sound_pop'
    )
    sound_pop.setVolume(1.0)
    sound_pop2 = sound.Sound(
        'A', 
        secs=1.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_pop2',    name='sound_pop2'
    )
    sound_pop2.setVolume(1.0)
    sound_pop3 = sound.Sound(
        'A', 
        secs=1.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_pop3',    name='sound_pop3'
    )
    sound_pop3.setVolume(1.0)
    
    # Unknown component ignored: MarkEvents_Encoding1_first_view
    
    
    # --- Initialize components for Routine "mask" ---
    Mask = visual.ImageStim(
        win=win,
        name='Mask', 
        image='images/Mask.png', mask=None, anchor='center',
        ori=0, pos=(0, 0), size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=0.0)
    
    # Unknown component ignored: MarkEvents_mask
    
    
    # --- Initialize components for Routine "second_view" ---
    img1 = visual.ImageStim(
        win=win,
        name='img1', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-1.0)
    img2 = visual.ImageStim(
        win=win,
        name='img2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-2.0)
    img3 = visual.ImageStim(
        win=win,
        name='img3', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-3.0)
    sound_pop4 = sound.Sound(
        'A', 
        secs=1.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_pop4',    name='sound_pop4'
    )
    sound_pop4.setVolume(1.0)
    sound_pop5 = sound.Sound(
        'A', 
        secs=1.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_pop5',    name='sound_pop5'
    )
    sound_pop5.setVolume(1.0)
    sound_pop6 = sound.Sound(
        'A', 
        secs=1.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_pop6',    name='sound_pop6'
    )
    sound_pop6.setVolume(1.0)
    
    # Unknown component ignored: MarkEvents_Encoding1_second_view
    
    
    # --- Initialize components for Routine "methodQText" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Was this set same or different as the previous set?',
        font='Arial',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    
    # Unknown component ignored: MarkEvents_methodQText
    
    
    # --- Initialize components for Routine "MethodCheck" ---
    text = visual.TextStim(win=win, name='text',
        text='Was this set same or different as the previous set?',
        font='Arial',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    sameText = visual.TextStim(win=win, name='sameText',
        text='',
        font='Arial',
        pos=(-0.2, -0.3), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    diffText = visual.TextStim(win=win, name='diffText',
        text='',
        font='Arial',
        pos=(0.2, -0.3), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-3.0);
    leftCL = visual.Rect(
        win=win, name='leftCL',
        width=(-0.2, -0.2)[0], height=(-0.2, -0.2)[1],
        ori=0, pos=(-.2, -.3), anchor='center',
        lineWidth=3,     colorSpace='rgb',  lineColor='white', fillColor=(1.0000, 1.0000, 1.0000),
        opacity=0.5, depth=-4.0, interpolate=True)
    rightCL = visual.Rect(
        win=win, name='rightCL',
        width=(0.2, 0.2)[0], height=(0.2, 0.2)[1],
        ori=0, pos=(0.2, -0.3), anchor='center',
        lineWidth=3,     colorSpace='rgb',  lineColor='white', fillColor=(1.0000, 1.0000, 1.0000),
        opacity=0.5, depth=-5.0, interpolate=True)
    MethodMouse = event.Mouse(win=win)
    x, y = [None, None]
    MethodMouse.mouseClock = core.Clock()
    
    # Unknown component ignored: MarkEvents_Encoding1_MethodCheck
    
    
    # --- Initialize components for Routine "stopRecording" ---
    
    # Unknown component ignored: StopRecord
    
    
    # --- Initialize components for Routine "break_1" ---
    BlockBreak = visual.TextStim(win=win, name='BlockBreak',
        text='You are done with the first portion of the task!\n\nNow, you have a three minute break. When the time is over, you will see instructions on the screen.',
        font='Arial',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    text_4 = visual.TextStim(win=win, name='text_4',
        text='You will be asked to remember when in the triplet you saw objects. Press the spacebar to begin.',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    start2 = keyboard.Keyboard(deviceName='start2')
    exp_endbreak = keyboard.Keyboard(deviceName='exp_endbreak')
    
    # --- Initialize components for Routine "initRecallPics" ---
    
    # --- Initialize components for Routine "eyeCheck" ---
    
    # Unknown component ignored: driftCheck
    
    
    # --- Initialize components for Routine "startRecordingRetrieval1" ---
    
    # Unknown component ignored: HostDrawing_Retrieval1
    
    
    # Unknown component ignored: StartRecord_Retrieval1
    
    
    # --- Initialize components for Routine "fixation" ---
    fix = visual.ImageStim(
        win=win,
        name='fix', 
        image='images/fixation.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.07, 0.05),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # Unknown component ignored: MarkEvents_fixation
    
    
    # --- Initialize components for Routine "Q2Text" ---
    question_text = visual.TextStim(win=win, name='question_text',
        text='When did you see the following object?',
        font='Arial',
        pos=(0, 0.3), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    
    # Unknown component ignored: MarkEvents_Q2Text
    
    
    # --- Initialize components for Routine "RetrievalImage" ---
    retIMG = visual.ImageStim(
        win=win,
        name='retIMG', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=(0, 0), size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-1.0)
    sound_pop7 = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_pop7',    name='sound_pop7'
    )
    sound_pop7.setVolume(1.0)
    
    # Unknown component ignored: MarkEvents_Retrieval1_RetrievalImage
    
    
    # --- Initialize components for Routine "Decision" ---
    decisionQ_2 = visual.TextStim(win=win, name='decisionQ_2',
        text='When did you see this object?',
        font='Arial',
        pos=(0, 0.3), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    first_left = visual.Rect(
        win=win, name='first_left',
        width=(0.15, 0.15)[0], height=(0.15, 0.15)[1],
        ori=0, pos=(0, 0.1), anchor='center',
        lineWidth=3,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=[1,1,1],
        opacity=0.5, depth=-2.0, interpolate=True)
    second_middle = visual.Rect(
        win=win, name='second_middle',
        width=(0.15, 0.15)[0], height=(0.15, 0.15)[1],
        ori=0, pos=(0, -0.1), anchor='center',
        lineWidth=3,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=[1,1,1],
        opacity=0.5, depth=-3.0, interpolate=True)
    third_right = visual.Rect(
        win=win, name='third_right',
        width=(0.15, 0.15)[0], height=(0.15, 0.15)[1],
        ori=0, pos=(0, -0.3), anchor='center',
        lineWidth=3,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=[1,1,1],
        opacity=0.5, depth=-4.0, interpolate=True)
    First = visual.TextStim(win=win, name='First',
        text='First',
        font='Arial',
        pos=(0, 0.1), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-5.0);
    Second = visual.TextStim(win=win, name='Second',
        text='Second',
        font='Arial',
        pos=(0, -0.1), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-6.0);
    Third = visual.TextStim(win=win, name='Third',
        text='Third',
        font='Arial',
        pos=(0, -0.3), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-7.0);
    Order_Position_Response = event.Mouse(win=win)
    x, y = [None, None]
    Order_Position_Response.mouseClock = core.Clock()
    
    # Unknown component ignored: MarkEvents_Retrieval1_Decision
    
    
    # --- Initialize components for Routine "stopRecording" ---
    
    # Unknown component ignored: StopRecord
    
    
    # --- Initialize components for Routine "init2" ---
    
    # --- Initialize components for Routine "trial" ---
    Instructions = visual.TextStim(win=win, name='Instructions',
        text='In this task you will review sets of objects in triplets. Your task is to pay attention to where and when these objects appear. Press the spacebar to begin.',
        font='Arial',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    start = keyboard.Keyboard(deviceName='start')
    
    # --- Initialize components for Routine "startRecordingEncoding2" ---
    
    # Unknown component ignored: HostDrawing_Encoding2
    
    
    # Unknown component ignored: StartRecord_Encoding2
    
    
    # --- Initialize components for Routine "fixation" ---
    fix = visual.ImageStim(
        win=win,
        name='fix', 
        image='images/fixation.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.07, 0.05),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # Unknown component ignored: MarkEvents_fixation
    
    
    # --- Initialize components for Routine "first_view2" ---
    Triplet1_2 = visual.ImageStim(
        win=win,
        name='Triplet1_2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-1.0)
    Triplet2_2 = visual.ImageStim(
        win=win,
        name='Triplet2_2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-2.0)
    Triplet3_2 = visual.ImageStim(
        win=win,
        name='Triplet3_2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-3.0)
    sound_pop_2 = sound.Sound(
        'A', 
        secs=1.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_pop_2',    name='sound_pop_2'
    )
    sound_pop_2.setVolume(1.0)
    sound_pop2_2 = sound.Sound(
        'A', 
        secs=1.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_pop2_2',    name='sound_pop2_2'
    )
    sound_pop2_2.setVolume(1.0)
    sound_pop3_2 = sound.Sound(
        'A', 
        secs=1.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_pop3_2',    name='sound_pop3_2'
    )
    sound_pop3_2.setVolume(1.0)
    
    # Unknown component ignored: MarkEvents_Encoding2_first_view2
    
    
    # --- Initialize components for Routine "mask" ---
    Mask = visual.ImageStim(
        win=win,
        name='Mask', 
        image='images/Mask.png', mask=None, anchor='center',
        ori=0, pos=(0, 0), size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=0.0)
    
    # Unknown component ignored: MarkEvents_mask
    
    
    # --- Initialize components for Routine "second_view2" ---
    img1_2 = visual.ImageStim(
        win=win,
        name='img1_2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-1.0)
    img2_2 = visual.ImageStim(
        win=win,
        name='img2_2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-2.0)
    img3_2 = visual.ImageStim(
        win=win,
        name='img3_2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[0,0], size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-3.0)
    sound_pop4_2 = sound.Sound(
        'A', 
        secs=1.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_pop4_2',    name='sound_pop4_2'
    )
    sound_pop4_2.setVolume(1.0)
    sound_pop5_2 = sound.Sound(
        'A', 
        secs=1.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_pop5_2',    name='sound_pop5_2'
    )
    sound_pop5_2.setVolume(1.0)
    sound_pop6_2 = sound.Sound(
        'A', 
        secs=1.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_pop6_2',    name='sound_pop6_2'
    )
    sound_pop6_2.setVolume(1.0)
    
    # Unknown component ignored: MarkEvents_Encoding2_second_view2
    
    
    # --- Initialize components for Routine "methodQText" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Was this set same or different as the previous set?',
        font='Arial',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    
    # Unknown component ignored: MarkEvents_methodQText
    
    
    # --- Initialize components for Routine "MethodCheck2" ---
    text_5 = visual.TextStim(win=win, name='text_5',
        text='Was this set same or different as the previous set?',
        font='Arial',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    sameText_2 = visual.TextStim(win=win, name='sameText_2',
        text='',
        font='Arial',
        pos=(-0.2, -0.3), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    diffText_2 = visual.TextStim(win=win, name='diffText_2',
        text='',
        font='Arial',
        pos=(0.2, -0.3), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-3.0);
    leftCL_2 = visual.Rect(
        win=win, name='leftCL_2',
        width=(-0.2, -0.2)[0], height=(-0.2, -0.2)[1],
        ori=0, pos=(-.2, -.3), anchor='center',
        lineWidth=3,     colorSpace='rgb',  lineColor='white', fillColor=(1.0000, 1.0000, 1.0000),
        opacity=0.5, depth=-4.0, interpolate=True)
    rightCL_2 = visual.Rect(
        win=win, name='rightCL_2',
        width=(0.2, 0.2)[0], height=(0.2, 0.2)[1],
        ori=0, pos=(0.2, -0.3), anchor='center',
        lineWidth=3,     colorSpace='rgb',  lineColor='white', fillColor=(1.0000, 1.0000, 1.0000),
        opacity=0.5, depth=-5.0, interpolate=True)
    MethodMouse_2 = event.Mouse(win=win)
    x, y = [None, None]
    MethodMouse_2.mouseClock = core.Clock()
    
    # Unknown component ignored: MarkEvents_Encoding2_MethodCheck2
    
    
    # --- Initialize components for Routine "stopRecording" ---
    
    # Unknown component ignored: StopRecord
    
    
    # --- Initialize components for Routine "break_1" ---
    BlockBreak = visual.TextStim(win=win, name='BlockBreak',
        text='You are done with the first portion of the task!\n\nNow, you have a three minute break. When the time is over, you will see instructions on the screen.',
        font='Arial',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    text_4 = visual.TextStim(win=win, name='text_4',
        text='You will be asked to remember when in the triplet you saw objects. Press the spacebar to begin.',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    start2 = keyboard.Keyboard(deviceName='start2')
    exp_endbreak = keyboard.Keyboard(deviceName='exp_endbreak')
    
    # --- Initialize components for Routine "initRecallPics2" ---
    
    # --- Initialize components for Routine "eyeCheck2" ---
    
    # Unknown component ignored: drift2
    
    
    # --- Initialize components for Routine "startRecording_Retrieval2" ---
    
    # Unknown component ignored: HostDrawing_Retrieval2
    
    
    # Unknown component ignored: StartRecord_Retrieval2
    
    
    # --- Initialize components for Routine "fixation" ---
    fix = visual.ImageStim(
        win=win,
        name='fix', 
        image='images/fixation.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.07, 0.05),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # Unknown component ignored: MarkEvents_fixation
    
    
    # --- Initialize components for Routine "Q2Text" ---
    question_text = visual.TextStim(win=win, name='question_text',
        text='When did you see the following object?',
        font='Arial',
        pos=(0, 0.3), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    
    # Unknown component ignored: MarkEvents_Q2Text
    
    
    # --- Initialize components for Routine "RetrievalImage2" ---
    retIMG_2 = visual.ImageStim(
        win=win,
        name='retIMG_2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=(0, 0), size=(0.25,0.25),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-1.0)
    sound_pop7_2 = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_pop7_2',    name='sound_pop7_2'
    )
    sound_pop7_2.setVolume(1.0)
    
    # Unknown component ignored: MarkEvents_Retrieval2
    
    
    # --- Initialize components for Routine "Decision2" ---
    decisionQ_3 = visual.TextStim(win=win, name='decisionQ_3',
        text='When did you see this object?',
        font='Arial',
        pos=(0, 0.3), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    first_left_2 = visual.Rect(
        win=win, name='first_left_2',
        width=(0.15, 0.15)[0], height=(0.15, 0.15)[1],
        ori=0, pos=(0, 0.1), anchor='center',
        lineWidth=3,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=[1,1,1],
        opacity=0.5, depth=-2.0, interpolate=True)
    second_middle_2 = visual.Rect(
        win=win, name='second_middle_2',
        width=(0.15, 0.15)[0], height=(0.15, 0.15)[1],
        ori=0, pos=(0, -0.1), anchor='center',
        lineWidth=3,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=[1,1,1],
        opacity=0.5, depth=-3.0, interpolate=True)
    third_right_2 = visual.Rect(
        win=win, name='third_right_2',
        width=(0.15, 0.15)[0], height=(0.15, 0.15)[1],
        ori=0, pos=(0, -0.3), anchor='center',
        lineWidth=3,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=[1,1,1],
        opacity=0.5, depth=-4.0, interpolate=True)
    First_2 = visual.TextStim(win=win, name='First_2',
        text='First',
        font='Arial',
        pos=(0, 0.1), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-5.0);
    Second_2 = visual.TextStim(win=win, name='Second_2',
        text='Second',
        font='Arial',
        pos=(0, -0.1), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-6.0);
    Third_2 = visual.TextStim(win=win, name='Third_2',
        text='Third',
        font='Arial',
        pos=(0, -0.3), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-7.0);
    Order_Position_Response_2 = event.Mouse(win=win)
    x, y = [None, None]
    Order_Position_Response_2.mouseClock = core.Clock()
    
    # Unknown component ignored: MarkEvents_Retrieval2_Decision2
    
    
    # --- Initialize components for Routine "stopRecording" ---
    
    # Unknown component ignored: StopRecord
    
    
    # --- Initialize components for Routine "EndScreen" ---
    end = visual.TextStim(win=win, name='end',
        text='You are done with this task! Thank you for participating!',
        font='Arial',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "eyelinkSetup" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('eyelinkSetup.started', globalClock.getTime(format='float'))
    el_key_resp.keys = []
    el_key_resp.rt = []
    _el_key_resp_allKeys = []
    # keep track of which components have finished
    eyelinkSetupComponents = [el_key_resp, Initialize, CameraSetup]
    for thisComponent in eyelinkSetupComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "eyelinkSetup" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *el_key_resp* updates
        waitOnFlip = False
        
        # if el_key_resp is starting this frame...
        if el_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            el_key_resp.frameNStart = frameN  # exact frame index
            el_key_resp.tStart = t  # local t and not account for scr refresh
            el_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(el_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'el_key_resp.started')
            # update status
            el_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(el_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(el_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if el_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = el_key_resp.getKeys(keyList=None, ignoreKeys=None, waitRelease=False)
            _el_key_resp_allKeys.extend(theseKeys)
            if len(_el_key_resp_allKeys):
                el_key_resp.keys = _el_key_resp_allKeys[-1].name  # just the last key pressed
                el_key_resp.rt = _el_key_resp_allKeys[-1].rt
                el_key_resp.duration = _el_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in eyelinkSetupComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "eyelinkSetup" ---
    for thisComponent in eyelinkSetupComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('eyelinkSetup.stopped', globalClock.getTime(format='float'))
    # check responses
    if el_key_resp.keys in ['', [], None]:  # No response was made
        el_key_resp.keys = None
    thisExp.addData('el_key_resp.keys',el_key_resp.keys)
    if el_key_resp.keys != None:  # we had a response
        thisExp.addData('el_key_resp.rt', el_key_resp.rt)
        thisExp.addData('el_key_resp.duration', el_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "eyelinkSetup" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "initCode" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('initCode.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from randomizeImages
    allPics = ['OBJ_1','OBJ_2','OBJ_3','OBJ_4','OBJ_5','OBJ_6','OBJ_7','OBJ_8','OBJ_9','OBJ_10','OBJ_11','OBJ_12','OBJ_13','OBJ_14','OBJ_15','OBJ_16','OBJ_17','OBJ_18','OBJ_19','OBJ_20','OBJ_21','OBJ_22','OBJ_23','OBJ_24','OBJ_25','OBJ_26','OBJ_28','OBJ_29','OBJ_30','OBJ_31','OBJ_32','OBJ_33','OBJ_35','OBJ_36','OBJ_37','OBJ_38','OBJ_39','OBJ_40','OBJ_41','OBJ_42','OBJ_43','OBJ_44','OBJ_45','OBJ_46','OBJ_47','OBJ_48','OBJ_49','OBJ_50','OBJ_51','OBJ_52','OBJ_53','OBJ_54','OBJ_55','OBJ_56','OBJ_57','OBJ_58','OBJ_59','OBJ_60','OBJ_61','OBJ_63','OBJ_64','OBJ_65','OBJ_66','OBJ_67','OBJ_68','OBJ_69','OBJ_70','OBJ_71','OBJ_72','OBJ_73','OBJ_74','OBJ_75','OBJ_77','OBJ_78','OBJ_79','OBJ_80','OBJ_81','OBJ_82','OBJ_83','OBJ_84','OBJ_85','OBJ_86','OBJ_87','OBJ_88','OBJ_89','OBJ_91','OBJ_92','OBJ_93','OBJ_94','OBJ_95','OBJ_96','OBJ_97','OBJ_98','OBJ_99','OBJ_100','OBJ_101','OBJ_102','OBJ_103','OBJ_104','OBJ_105','OBJ_106','OBJ_107','OBJ_109','OBJ_110','OBJ_111','OBJ_112','OBJ_113','OBJ_114','OBJ_115','OBJ_116','OBJ_119','OBJ_120','OBJ_121','OBJ_123','OBJ_125','OBJ_126','OBJ_128','OBJ_129','OBJ_130','OBJ_131','OBJ_133','OBJ_134','OBJ_135','OBJ_136','OBJ_137','OBJ_138','OBJ_139','OBJ_140','OBJ_141','OBJ_142','OBJ_143','OBJ_144','OBJ_145','OBJ_146','OBJ_147','OBJ_148','OBJ_149','OBJ_150','OBJ_151','OBJ_152','OBJ_153','OBJ_154','OBJ_155','OBJ_156','OBJ_157','OBJ_158','OBJ_160','OBJ_161','OBJ_163','OBJ_164','OBJ_165','OBJ_166','OBJ_167','OBJ_168','OBJ_170','OBJ_171','OBJ_172','OBJ_173','OBJ_174','OBJ_175','OBJ_177','OBJ_178','OBJ_179','OBJ_181','OBJ_182','OBJ_183','OBJ_184','OBJ_186','OBJ_187','OBJ_188','OBJ_189','OBJ_190','OBJ_191','OBJ_193','OBJ_194','OBJ_195','OBJ_196','OBJ_197','OBJ_198','OBJ_199','OBJ_200','OBJ_202','OBJ_203','OBJ_204','OBJ_205','OBJ_206','OBJ_208','OBJ_209','OBJ_210'];
    sameDiff = ["same","same","same","same","same","same","same","same","same","same","same","same","same","same","same","same","same","same","same","same","diff","diff","diff","diff","diff"];
    shuffle(allPics)
    shuffle(sameDiff)
    
    #So for the same trials:
    # Define the strings
    fixed_strings = ['LMR', 'RML']  # These will occur 7 times each
    other_strings = ['MLR', 'MRL', 'LRM', 'RLM']  # These will be randomly selected to have total of 7 also
    
    # Select strings from other_strings with repetitions
    selected_strings = random.choices(other_strings, k=9)
    
    # Create a list to store the strings
    sameTrials = fixed_strings * 9  # Repeating fixed_strings 9 times
    sameTrials += selected_strings  # Adding the selected 9 strings
    
    # Shuffle the list
    random.shuffle(sameTrials)
    
    # Print the randomized list
    print("Randomized string list:")
    print(sameTrials) #now this should be a list of 27 that we will run 20 of, but because it is indexed by eTrials, there needs to be more than 25 items. 
    
    #for the different trials:
    diffTrials = fixed_strings * 9  # Repeating fixed_strings 9 times
    diffTrials += selected_strings  # Adding the selected 9 strings
    
    # Shuffle the list
    random.shuffle(diffTrials)
    
    # Print the selected strings)
    print("'different' trials:")
    print(diffTrials) #because these are indexed by eTrials while they select the images, they need to be more than 25 items
    #hard-coded names of all images and trial types (counterbalanced).
    #could change these to have items come from a csv at this point.
    
    numItems = 210
    numEncodingBlock1 = 2
    numEncodingBlock2 = 25
    numTotretrieval = 40
    numRetrieval1 = 1
    numRetrieval2 = 20
    
    # create final stimuli list
    for i in range(0, numItems, 3):
        # make triplet
        trialPics.append(allPics[i:i+3])
        diffTrials += selected_strings
    #70 triplets made from 210 images.
    
    # these triplets will be shown during encoding
    # Create indices for selecting triplets for each block
    indices_block1 = random.sample(range(len(allPics) // 3), numEncodingBlock1)
    indices_block2 = random.sample(set(range(len(allPics) // 3)) - set(indices_block1), numEncodingBlock2)
    
    # Create encodingPics for each block using the selected indices
    encodingPics1 = [allPics[i * 3:i * 3 + 3] for i in indices_block1]
    encodingPics2 = [allPics[i * 3:i * 3 + 3] for i in indices_block2]
    
    # Now you have unique sets of triplets for each block
    print(encodingPics1)
    print(encodingPics2)
    
    #this is where I will store the images from same and diff trials so we know which ones to use for retrieval
    samePics = []
    newPics = []
    samePics2 = [] #for block 2
    newPics2 = [] #for block 2
    
    #set trial num
    eTrials = 0
    TRIAL_INDEX = 0
    # keep track of which components have finished
    initCodeComponents = []
    for thisComponent in initCodeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "initCode" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in initCodeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "initCode" ---
    for thisComponent in initCodeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('initCode.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    # the Routine "initCode" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "trial" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('trial.started', globalClock.getTime(format='float'))
    start.keys = []
    start.rt = []
    _start_allKeys = []
    # keep track of which components have finished
    trialComponents = [Instructions, start]
    for thisComponent in trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "trial" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Instructions* updates
        
        # if Instructions is starting this frame...
        if Instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Instructions.frameNStart = frameN  # exact frame index
            Instructions.tStart = t  # local t and not account for scr refresh
            Instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Instructions.started')
            # update status
            Instructions.status = STARTED
            Instructions.setAutoDraw(True)
        
        # if Instructions is active this frame...
        if Instructions.status == STARTED:
            # update params
            pass
        
        # *start* updates
        waitOnFlip = False
        
        # if start is starting this frame...
        if start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start.frameNStart = frameN  # exact frame index
            start.tStart = t  # local t and not account for scr refresh
            start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start.started')
            # update status
            start.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start.clock.reset)  # t=0 on next screen flip
        if start.status == STARTED and not waitOnFlip:
            theseKeys = start.getKeys(keyList=['space','q'], ignoreKeys=None, waitRelease=False)
            _start_allKeys.extend(theseKeys)
            if len(_start_allKeys):
                start.keys = _start_allKeys[-1].name  # just the last key pressed
                start.rt = _start_allKeys[-1].rt
                start.duration = _start_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "trial" ---
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('trial.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    # the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    EncodingTrials = data.TrialHandler(nReps=numEncodingBlock1, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='EncodingTrials')
    thisExp.addLoop(EncodingTrials)  # add the loop to the experiment
    thisEncodingTrial = EncodingTrials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisEncodingTrial.rgb)
    if thisEncodingTrial != None:
        for paramName in thisEncodingTrial:
            globals()[paramName] = thisEncodingTrial[paramName]
    
    for thisEncodingTrial in EncodingTrials:
        currentLoop = EncodingTrials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisEncodingTrial.rgb)
        if thisEncodingTrial != None:
            for paramName in thisEncodingTrial:
                globals()[paramName] = thisEncodingTrial[paramName]
        
        # --- Prepare to start Routine "startRecordingEncoding1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('startRecordingEncoding1.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        startRecordingEncoding1Components = [HostDrawing_Encoding1, StartRecord_Endcoding1]
        for thisComponent in startRecordingEncoding1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "startRecordingEncoding1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.001:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in startRecordingEncoding1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "startRecordingEncoding1" ---
        for thisComponent in startRecordingEncoding1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('startRecordingEncoding1.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.001000)
        
        # --- Prepare to start Routine "fixation" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fixation.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        fixationComponents = [fix, MarkEvents_fixation]
        for thisComponent in fixationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fix* updates
            
            # if fix is starting this frame...
            if fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix.frameNStart = frameN  # exact frame index
                fix.tStart = t  # local t and not account for scr refresh
                fix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix.started')
                # update status
                fix.status = STARTED
                fix.setAutoDraw(True)
            
            # if fix is active this frame...
            if fix.status == STARTED:
                # update params
                pass
            
            # if fix is stopping this frame...
            if fix.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fix.tStop = t  # not accounting for scr refresh
                    fix.tStopRefresh = tThisFlipGlobal  # on global time
                    fix.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix.stopped')
                    # update status
                    fix.status = FINISHED
                    fix.setAutoDraw(False)
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('fixation.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "first_view" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('first_view.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from firstw_code
        
        #iterate through trial type and set a new one for each trial
        #trialType = trialTypes[eTrials]
        #method_Trial = sameDiff[eTrials] 
        
        #iterate through trial type and set a new one for each trial
        method_Trial = sameDiff[eTrials]
        if method_Trial == "same":
            trialType = sameTrials[eTrials]
        else:
            trialType = diffTrials[eTrials]
            
        #positions
        leftPos = -0.5, 0.0
        midPos = 0.0, 0.0
        rightPos = 0.5, 0.0
            
        if trialType == 'LMR':
            firstPos = leftPos
            secondPos = midPos
            thirdPos = rightPos
        elif trialType == 'LRM':
                firstPos = leftPos
                secondPos = rightPos
                thirdPos = midPos
        elif trialType == 'MLR':
                firstPos = midPos
                secondPos = leftPos
                thirdPos = rightPos
        elif trialType == 'RML':
                firstPos = rightPos
                secondPos = midPos
                thirdPos = leftPos
        elif trialType == 'RLM':
                firstPos = rightPos
                secondPos = leftPos
                thirdPos = midPos
        elif trialType == 'MRL':
                firstPos = midPos
                secondPos = rightPos
                thirdPos = leftPos
            
        #set images in triplets
        triplet1 = 'images/'+encodingPics1[eTrials][0]+'.png'
        triplet2 = 'images/'+encodingPics1[eTrials][1]+'.png'
        triplet3 = 'images/'+encodingPics1[eTrials][2]+'.png'
        
        firstPos1 = firstPos
        secondPos1 = secondPos
        thirdPos1 = thirdPos
        triplet1_1 = triplet1
        triplet2_1 = triplet2
        triplet3_1 = triplet3 
        
        EncodingTrials.addData ("Trial_Order", trialType)
        thisExp.addData("triplet1", triplet1)
        thisExp.addData("triplet2", triplet2)
        thisExp.addData("triplet3", triplet3)
        trial_category = "encoding"
        Triplet1.setPos(firstPos)
        Triplet1.setImage(triplet1)
        Triplet2.setPos(secondPos)
        Triplet2.setImage(triplet2)
        Triplet3.setPos(thirdPos)
        Triplet3.setImage(triplet3)
        sound_pop.setSound('POP.wav', secs=1.5, hamming=True)
        sound_pop.setVolume(1.0, log=False)
        sound_pop.seek(0)
        sound_pop2.setSound('POP.wav', secs=1.5, hamming=True)
        sound_pop2.setVolume(1.0, log=False)
        sound_pop2.seek(0)
        sound_pop3.setSound('POP.wav', secs=1.5, hamming=True)
        sound_pop3.setVolume(1.0, log=False)
        sound_pop3.seek(0)
        # keep track of which components have finished
        first_viewComponents = [Triplet1, Triplet2, Triplet3, sound_pop, sound_pop2, sound_pop3, MarkEvents_Encoding1_first_view]
        for thisComponent in first_viewComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "first_view" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Triplet1* updates
            
            # if Triplet1 is starting this frame...
            if Triplet1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Triplet1.frameNStart = frameN  # exact frame index
                Triplet1.tStart = t  # local t and not account for scr refresh
                Triplet1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Triplet1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Triplet1.started')
                # update status
                Triplet1.status = STARTED
                Triplet1.setAutoDraw(True)
            
            # if Triplet1 is active this frame...
            if Triplet1.status == STARTED:
                # update params
                pass
            
            # if Triplet1 is stopping this frame...
            if Triplet1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Triplet1.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Triplet1.tStop = t  # not accounting for scr refresh
                    Triplet1.tStopRefresh = tThisFlipGlobal  # on global time
                    Triplet1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Triplet1.stopped')
                    # update status
                    Triplet1.status = FINISHED
                    Triplet1.setAutoDraw(False)
            
            # *Triplet2* updates
            
            # if Triplet2 is starting this frame...
            if Triplet2.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                Triplet2.frameNStart = frameN  # exact frame index
                Triplet2.tStart = t  # local t and not account for scr refresh
                Triplet2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Triplet2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Triplet2.started')
                # update status
                Triplet2.status = STARTED
                Triplet2.setAutoDraw(True)
            
            # if Triplet2 is active this frame...
            if Triplet2.status == STARTED:
                # update params
                pass
            
            # if Triplet2 is stopping this frame...
            if Triplet2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Triplet2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Triplet2.tStop = t  # not accounting for scr refresh
                    Triplet2.tStopRefresh = tThisFlipGlobal  # on global time
                    Triplet2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Triplet2.stopped')
                    # update status
                    Triplet2.status = FINISHED
                    Triplet2.setAutoDraw(False)
            
            # *Triplet3* updates
            
            # if Triplet3 is starting this frame...
            if Triplet3.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                Triplet3.frameNStart = frameN  # exact frame index
                Triplet3.tStart = t  # local t and not account for scr refresh
                Triplet3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Triplet3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Triplet3.started')
                # update status
                Triplet3.status = STARTED
                Triplet3.setAutoDraw(True)
            
            # if Triplet3 is active this frame...
            if Triplet3.status == STARTED:
                # update params
                pass
            
            # if Triplet3 is stopping this frame...
            if Triplet3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Triplet3.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Triplet3.tStop = t  # not accounting for scr refresh
                    Triplet3.tStopRefresh = tThisFlipGlobal  # on global time
                    Triplet3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Triplet3.stopped')
                    # update status
                    Triplet3.status = FINISHED
                    Triplet3.setAutoDraw(False)
            
            # if sound_pop is starting this frame...
            if sound_pop.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                sound_pop.frameNStart = frameN  # exact frame index
                sound_pop.tStart = t  # local t and not account for scr refresh
                sound_pop.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_pop.started', tThisFlipGlobal)
                # update status
                sound_pop.status = STARTED
                sound_pop.play(when=win)  # sync with win flip
            
            # if sound_pop is stopping this frame...
            if sound_pop.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_pop.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_pop.tStop = t  # not accounting for scr refresh
                    sound_pop.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_pop.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_pop.stopped')
                    # update status
                    sound_pop.status = FINISHED
                    sound_pop.stop()
            # update sound_pop status according to whether it's playing
            if sound_pop.isPlaying:
                sound_pop.status = STARTED
            elif sound_pop.isFinished:
                sound_pop.status = FINISHED
            
            # if sound_pop2 is starting this frame...
            if sound_pop2.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                sound_pop2.frameNStart = frameN  # exact frame index
                sound_pop2.tStart = t  # local t and not account for scr refresh
                sound_pop2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_pop2.started', tThisFlipGlobal)
                # update status
                sound_pop2.status = STARTED
                sound_pop2.play(when=win)  # sync with win flip
            
            # if sound_pop2 is stopping this frame...
            if sound_pop2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_pop2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_pop2.tStop = t  # not accounting for scr refresh
                    sound_pop2.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_pop2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_pop2.stopped')
                    # update status
                    sound_pop2.status = FINISHED
                    sound_pop2.stop()
            # update sound_pop2 status according to whether it's playing
            if sound_pop2.isPlaying:
                sound_pop2.status = STARTED
            elif sound_pop2.isFinished:
                sound_pop2.status = FINISHED
            
            # if sound_pop3 is starting this frame...
            if sound_pop3.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                sound_pop3.frameNStart = frameN  # exact frame index
                sound_pop3.tStart = t  # local t and not account for scr refresh
                sound_pop3.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_pop3.started', tThisFlipGlobal)
                # update status
                sound_pop3.status = STARTED
                sound_pop3.play(when=win)  # sync with win flip
            
            # if sound_pop3 is stopping this frame...
            if sound_pop3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_pop3.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_pop3.tStop = t  # not accounting for scr refresh
                    sound_pop3.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_pop3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_pop3.stopped')
                    # update status
                    sound_pop3.status = FINISHED
                    sound_pop3.stop()
            # update sound_pop3 status according to whether it's playing
            if sound_pop3.isPlaying:
                sound_pop3.status = STARTED
            elif sound_pop3.isFinished:
                sound_pop3.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in first_viewComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "first_view" ---
        for thisComponent in first_viewComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('first_view.stopped', globalClock.getTime(format='float'))
        sound_pop.pause()  # ensure sound has stopped at end of Routine
        sound_pop2.pause()  # ensure sound has stopped at end of Routine
        sound_pop3.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.500000)
        
        # --- Prepare to start Routine "mask" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('mask.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        maskComponents = [Mask, MarkEvents_mask]
        for thisComponent in maskComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "mask" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Mask* updates
            
            # if Mask is starting this frame...
            if Mask.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Mask.frameNStart = frameN  # exact frame index
                Mask.tStart = t  # local t and not account for scr refresh
                Mask.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Mask, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Mask.started')
                # update status
                Mask.status = STARTED
                Mask.setAutoDraw(True)
            
            # if Mask is active this frame...
            if Mask.status == STARTED:
                # update params
                pass
            
            # if Mask is stopping this frame...
            if Mask.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Mask.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    Mask.tStop = t  # not accounting for scr refresh
                    Mask.tStopRefresh = tThisFlipGlobal  # on global time
                    Mask.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Mask.stopped')
                    # update status
                    Mask.status = FINISHED
                    Mask.setAutoDraw(False)
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in maskComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "mask" ---
        for thisComponent in maskComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('mask.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "second_view" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('second_view.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from secondw_code
        # update component parameters for each repeat
        if method_Trial == "same":
            list = [0,1,2]
            random.shuffle(list)
            samePics.append(encodingPics1[eTrials][list[0]])
            EncodingTrials.addData("encodingType","same")
            encodingType = 'same'
            methodAns = ['leftCL'] #same
        else:
            nextTrial = random.choice(diffTrials)
            while nextTrial == trialType:
                nextTrial = random.choice(diffTrials)
                end
            trialType = nextTrial
            methodAns = ['rightCL'] #different
            list = [0,1,2]
            random.shuffle(list)
            triplet1 = 'images/'+encodingPics1[eTrials][list[0]]+'.png'
            triplet2 = 'images/'+encodingPics1[eTrials][list[1]]+'.png'
            triplet3 = 'images/'+encodingPics1[eTrials][list[2]]+'.png'
            EncodingTrials.addData("encodingType","diff")
            encodingType = 'diff'
        
        #have to re-state positions and trial types
        if trialType == 'LMR':
            firstPos = leftPos
            secondPos = midPos
            thirdPos = rightPos
        elif trialType == 'LRM':
            firstPos = leftPos
            secondPos = rightPos
            thirdPos = midPos
        elif trialType == 'MLR':
            firstPos = midPos
            secondPos = leftPos
            thirdPos = rightPos
        elif trialType == 'RML':
            firstPos = rightPos
            secondPos = midPos
            thirdPos = leftPos
        elif trialType == 'RLM':
            firstPos = rightPos
            secondPos = leftPos
            thirdPos = midPos
        elif trialType == 'MRL':
            firstPos = midPos
            secondPos = rightPos
            thirdPos = leftPos
        
        firstPos2 = firstPos
        secondPos2 = secondPos
        thirdPos2 = thirdPos
        triplet1_2 = triplet1
        triplet2_2 = triplet2
        triplet3_2 = triplet3 
        
        EncodingTrials.addData("triplet1", triplet1)
        EncodingTrials.addData("triplet2", triplet2)
        EncodingTrials.addData("triplet3", triplet3)
        EncodingTrials.addData("nextTrial", nextTrial)
        img1.setPos(firstPos)
        img1.setImage(triplet1)
        img2.setPos(secondPos)
        img2.setImage(triplet2)
        img3.setPos(thirdPos)
        img3.setImage(triplet3)
        sound_pop4.setSound('POP.wav', secs=1.5, hamming=True)
        sound_pop4.setVolume(1.0, log=False)
        sound_pop4.seek(0)
        sound_pop5.setSound('POP.wav', secs=1.5, hamming=True)
        sound_pop5.setVolume(1.0, log=False)
        sound_pop5.seek(0)
        sound_pop6.setSound('POP.wav', secs=1.5, hamming=True)
        sound_pop6.setVolume(1.0, log=False)
        sound_pop6.seek(0)
        # keep track of which components have finished
        second_viewComponents = [img1, img2, img3, sound_pop4, sound_pop5, sound_pop6, MarkEvents_Encoding1_second_view]
        for thisComponent in second_viewComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "second_view" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *img1* updates
            
            # if img1 is starting this frame...
            if img1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                img1.frameNStart = frameN  # exact frame index
                img1.tStart = t  # local t and not account for scr refresh
                img1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img1.started')
                # update status
                img1.status = STARTED
                img1.setAutoDraw(True)
            
            # if img1 is active this frame...
            if img1.status == STARTED:
                # update params
                pass
            
            # if img1 is stopping this frame...
            if img1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > img1.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    img1.tStop = t  # not accounting for scr refresh
                    img1.tStopRefresh = tThisFlipGlobal  # on global time
                    img1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'img1.stopped')
                    # update status
                    img1.status = FINISHED
                    img1.setAutoDraw(False)
            
            # *img2* updates
            
            # if img2 is starting this frame...
            if img2.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                img2.frameNStart = frameN  # exact frame index
                img2.tStart = t  # local t and not account for scr refresh
                img2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img2.started')
                # update status
                img2.status = STARTED
                img2.setAutoDraw(True)
            
            # if img2 is active this frame...
            if img2.status == STARTED:
                # update params
                pass
            
            # if img2 is stopping this frame...
            if img2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > img2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    img2.tStop = t  # not accounting for scr refresh
                    img2.tStopRefresh = tThisFlipGlobal  # on global time
                    img2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'img2.stopped')
                    # update status
                    img2.status = FINISHED
                    img2.setAutoDraw(False)
            
            # *img3* updates
            
            # if img3 is starting this frame...
            if img3.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                img3.frameNStart = frameN  # exact frame index
                img3.tStart = t  # local t and not account for scr refresh
                img3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img3.started')
                # update status
                img3.status = STARTED
                img3.setAutoDraw(True)
            
            # if img3 is active this frame...
            if img3.status == STARTED:
                # update params
                pass
            
            # if img3 is stopping this frame...
            if img3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > img3.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    img3.tStop = t  # not accounting for scr refresh
                    img3.tStopRefresh = tThisFlipGlobal  # on global time
                    img3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'img3.stopped')
                    # update status
                    img3.status = FINISHED
                    img3.setAutoDraw(False)
            
            # if sound_pop4 is starting this frame...
            if sound_pop4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_pop4.frameNStart = frameN  # exact frame index
                sound_pop4.tStart = t  # local t and not account for scr refresh
                sound_pop4.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_pop4.started', tThisFlipGlobal)
                # update status
                sound_pop4.status = STARTED
                sound_pop4.play(when=win)  # sync with win flip
            
            # if sound_pop4 is stopping this frame...
            if sound_pop4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_pop4.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_pop4.tStop = t  # not accounting for scr refresh
                    sound_pop4.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_pop4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_pop4.stopped')
                    # update status
                    sound_pop4.status = FINISHED
                    sound_pop4.stop()
            # update sound_pop4 status according to whether it's playing
            if sound_pop4.isPlaying:
                sound_pop4.status = STARTED
            elif sound_pop4.isFinished:
                sound_pop4.status = FINISHED
            
            # if sound_pop5 is starting this frame...
            if sound_pop5.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                sound_pop5.frameNStart = frameN  # exact frame index
                sound_pop5.tStart = t  # local t and not account for scr refresh
                sound_pop5.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_pop5.started', tThisFlipGlobal)
                # update status
                sound_pop5.status = STARTED
                sound_pop5.play(when=win)  # sync with win flip
            
            # if sound_pop5 is stopping this frame...
            if sound_pop5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_pop5.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_pop5.tStop = t  # not accounting for scr refresh
                    sound_pop5.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_pop5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_pop5.stopped')
                    # update status
                    sound_pop5.status = FINISHED
                    sound_pop5.stop()
            # update sound_pop5 status according to whether it's playing
            if sound_pop5.isPlaying:
                sound_pop5.status = STARTED
            elif sound_pop5.isFinished:
                sound_pop5.status = FINISHED
            
            # if sound_pop6 is starting this frame...
            if sound_pop6.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                sound_pop6.frameNStart = frameN  # exact frame index
                sound_pop6.tStart = t  # local t and not account for scr refresh
                sound_pop6.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_pop6.started', tThisFlipGlobal)
                # update status
                sound_pop6.status = STARTED
                sound_pop6.play(when=win)  # sync with win flip
            
            # if sound_pop6 is stopping this frame...
            if sound_pop6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_pop6.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_pop6.tStop = t  # not accounting for scr refresh
                    sound_pop6.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_pop6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_pop6.stopped')
                    # update status
                    sound_pop6.status = FINISHED
                    sound_pop6.stop()
            # update sound_pop6 status according to whether it's playing
            if sound_pop6.isPlaying:
                sound_pop6.status = STARTED
            elif sound_pop6.isFinished:
                sound_pop6.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in second_viewComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "second_view" ---
        for thisComponent in second_viewComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('second_view.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from secondw_code
        eTrials = eTrials + 1
        TRIAL_INDEX = TRIAL_INDEX + 1
        
        #for psychopy
        EncodingTrials.addData("encodingTrials", str(eTrials))
        EncodingTrials.addData("TRIAL_INDEX", str(TRIAL_INDEX))
        
        #for EDF file
        el_tracker.sendMessage('!V TRIAL_VAR TRIAL_INDEX %d' % TRIAL_INDEX)
        el_tracker.sendMessage('!V TRIAL_VAR eTrials %d' %eTrials)
        sound_pop4.pause()  # ensure sound has stopped at end of Routine
        sound_pop5.pause()  # ensure sound has stopped at end of Routine
        sound_pop6.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.500000)
        
        # --- Prepare to start Routine "methodQText" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('methodQText.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        methodQTextComponents = [text_2, MarkEvents_methodQText]
        for thisComponent in methodQTextComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "methodQText" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.started')
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # if text_2 is stopping this frame...
            if text_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_2.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_2.tStop = t  # not accounting for scr refresh
                    text_2.tStopRefresh = tThisFlipGlobal  # on global time
                    text_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_2.stopped')
                    # update status
                    text_2.status = FINISHED
                    text_2.setAutoDraw(False)
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in methodQTextComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "methodQText" ---
        for thisComponent in methodQTextComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('methodQText.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "MethodCheck" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('MethodCheck.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from methodc_code
        win.mouseVisible = True
        sameText.setText('Same')
        diffText.setText('Different')
        # setup some python lists for storing info about the MethodMouse
        MethodMouse.x = []
        MethodMouse.y = []
        MethodMouse.leftButton = []
        MethodMouse.midButton = []
        MethodMouse.rightButton = []
        MethodMouse.time = []
        MethodMouse.clicked_name = []
        gotValidClick = False  # until a click is received
        # keep track of which components have finished
        MethodCheckComponents = [text, sameText, diffText, leftCL, rightCL, MethodMouse, MarkEvents_Encoding1_MethodCheck]
        for thisComponent in MethodCheckComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "MethodCheck" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # *sameText* updates
            
            # if sameText is starting this frame...
            if sameText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sameText.frameNStart = frameN  # exact frame index
                sameText.tStart = t  # local t and not account for scr refresh
                sameText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sameText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sameText.started')
                # update status
                sameText.status = STARTED
                sameText.setAutoDraw(True)
            
            # if sameText is active this frame...
            if sameText.status == STARTED:
                # update params
                pass
            
            # *diffText* updates
            
            # if diffText is starting this frame...
            if diffText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                diffText.frameNStart = frameN  # exact frame index
                diffText.tStart = t  # local t and not account for scr refresh
                diffText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(diffText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'diffText.started')
                # update status
                diffText.status = STARTED
                diffText.setAutoDraw(True)
            
            # if diffText is active this frame...
            if diffText.status == STARTED:
                # update params
                pass
            
            # *leftCL* updates
            
            # if leftCL is starting this frame...
            if leftCL.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                leftCL.frameNStart = frameN  # exact frame index
                leftCL.tStart = t  # local t and not account for scr refresh
                leftCL.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(leftCL, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'leftCL.started')
                # update status
                leftCL.status = STARTED
                leftCL.setAutoDraw(True)
            
            # if leftCL is active this frame...
            if leftCL.status == STARTED:
                # update params
                pass
            
            # *rightCL* updates
            
            # if rightCL is starting this frame...
            if rightCL.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rightCL.frameNStart = frameN  # exact frame index
                rightCL.tStart = t  # local t and not account for scr refresh
                rightCL.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rightCL, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rightCL.started')
                # update status
                rightCL.status = STARTED
                rightCL.setAutoDraw(True)
            
            # if rightCL is active this frame...
            if rightCL.status == STARTED:
                # update params
                pass
            # *MethodMouse* updates
            
            # if MethodMouse is starting this frame...
            if MethodMouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                MethodMouse.frameNStart = frameN  # exact frame index
                MethodMouse.tStart = t  # local t and not account for scr refresh
                MethodMouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(MethodMouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('MethodMouse.started', t)
                # update status
                MethodMouse.status = STARTED
                MethodMouse.mouseClock.reset()
                prevButtonState = MethodMouse.getPressed()  # if button is down already this ISN'T a new click
            if MethodMouse.status == STARTED:  # only update if started and not finished!
                buttons = MethodMouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([rightCL, leftCL], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(MethodMouse):
                                gotValidClick = True
                                MethodMouse.clicked_name.append(obj.name)
                        x, y = MethodMouse.getPos()
                        MethodMouse.x.append(x)
                        MethodMouse.y.append(y)
                        buttons = MethodMouse.getPressed()
                        MethodMouse.leftButton.append(buttons[0])
                        MethodMouse.midButton.append(buttons[1])
                        MethodMouse.rightButton.append(buttons[2])
                        MethodMouse.time.append(MethodMouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in MethodCheckComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "MethodCheck" ---
        for thisComponent in MethodCheckComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('MethodCheck.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from methodc_code
        #store whether or not they answered correctly for the "same" or "different" trial type.
        thisExp.addData("Method Clicked", MethodMouse.clicked_name);
        
        if MethodMouse.clicked_name == methodAns:
            EncodingTrials.addData ("methodCheck", '1')
            methodCheck = 1
        else:
            EncodingTrials.addData ("methodCheck", '0')
            methodCheck = 0
           
        #for psychopy
        EncodingTrials.addData("TRIAL_INDEX", str(TRIAL_INDEX))
        EncodingTrials.addData("trialCategory", str(trial_category))
        EncodingTrials.addData("encodingTrials", str(eTrials))
        EncodingTrials.addData("methodCheck", str(methodCheck))
        
        #for the EDF output
        el_tracker.sendMessage('!V TRIAL_VAR methodCheck %s' % methodCheck)
        el_tracker.sendMessage('!V TRIAL_VAR methodResp %s' % MethodMouse.clicked_name)
        
        win.mouseVisible = False
        # store data for EncodingTrials (TrialHandler)
        EncodingTrials.addData('MethodMouse.x', MethodMouse.x)
        EncodingTrials.addData('MethodMouse.y', MethodMouse.y)
        EncodingTrials.addData('MethodMouse.leftButton', MethodMouse.leftButton)
        EncodingTrials.addData('MethodMouse.midButton', MethodMouse.midButton)
        EncodingTrials.addData('MethodMouse.rightButton', MethodMouse.rightButton)
        EncodingTrials.addData('MethodMouse.time', MethodMouse.time)
        EncodingTrials.addData('MethodMouse.clicked_name', MethodMouse.clicked_name)
        # the Routine "MethodCheck" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "stopRecording" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('stopRecording.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        stopRecordingComponents = [StopRecord]
        for thisComponent in stopRecordingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "stopRecording" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.001:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in stopRecordingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "stopRecording" ---
        for thisComponent in stopRecordingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('stopRecording.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.001000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed numEncodingBlock1 repeats of 'EncodingTrials'
    
    
    # --- Prepare to start Routine "break_1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('break_1.started', globalClock.getTime(format='float'))
    start2.keys = []
    start2.rt = []
    _start2_allKeys = []
    exp_endbreak.keys = []
    exp_endbreak.rt = []
    _exp_endbreak_allKeys = []
    # keep track of which components have finished
    break_1Components = [BlockBreak, text_4, start2, exp_endbreak]
    for thisComponent in break_1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *BlockBreak* updates
        
        # if BlockBreak is starting this frame...
        if BlockBreak.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            BlockBreak.frameNStart = frameN  # exact frame index
            BlockBreak.tStart = t  # local t and not account for scr refresh
            BlockBreak.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(BlockBreak, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'BlockBreak.started')
            # update status
            BlockBreak.status = STARTED
            BlockBreak.setAutoDraw(True)
        
        # if BlockBreak is active this frame...
        if BlockBreak.status == STARTED:
            # update params
            pass
        
        # if BlockBreak is stopping this frame...
        if BlockBreak.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > BlockBreak.tStartRefresh + 180-frameTolerance:
                # keep track of stop time/frame for later
                BlockBreak.tStop = t  # not accounting for scr refresh
                BlockBreak.tStopRefresh = tThisFlipGlobal  # on global time
                BlockBreak.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'BlockBreak.stopped')
                # update status
                BlockBreak.status = FINISHED
                BlockBreak.setAutoDraw(False)
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 180.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_4.started')
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # *start2* updates
        waitOnFlip = False
        
        # if start2 is starting this frame...
        if start2.status == NOT_STARTED and tThisFlip >= 180.0-frameTolerance:
            # keep track of start time/frame for later
            start2.frameNStart = frameN  # exact frame index
            start2.tStart = t  # local t and not account for scr refresh
            start2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start2.started')
            # update status
            start2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(start2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if start2.status == STARTED and not waitOnFlip:
            theseKeys = start2.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _start2_allKeys.extend(theseKeys)
            if len(_start2_allKeys):
                start2.keys = _start2_allKeys[-1].name  # just the last key pressed
                start2.rt = _start2_allKeys[-1].rt
                start2.duration = _start2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *exp_endbreak* updates
        waitOnFlip = False
        
        # if exp_endbreak is starting this frame...
        if exp_endbreak.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            exp_endbreak.frameNStart = frameN  # exact frame index
            exp_endbreak.tStart = t  # local t and not account for scr refresh
            exp_endbreak.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(exp_endbreak, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'exp_endbreak.started')
            # update status
            exp_endbreak.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(exp_endbreak.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(exp_endbreak.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if exp_endbreak.status == STARTED and not waitOnFlip:
            theseKeys = exp_endbreak.getKeys(keyList=['z'], ignoreKeys=None, waitRelease=False)
            _exp_endbreak_allKeys.extend(theseKeys)
            if len(_exp_endbreak_allKeys):
                exp_endbreak.keys = _exp_endbreak_allKeys[-1].name  # just the last key pressed
                exp_endbreak.rt = _exp_endbreak_allKeys[-1].rt
                exp_endbreak.duration = _exp_endbreak_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_1" ---
    for thisComponent in break_1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('break_1.stopped', globalClock.getTime(format='float'))
    # check responses
    if start2.keys in ['', [], None]:  # No response was made
        start2.keys = None
    thisExp.addData('start2.keys',start2.keys)
    if start2.keys != None:  # we had a response
        thisExp.addData('start2.rt', start2.rt)
        thisExp.addData('start2.duration', start2.duration)
    # check responses
    if exp_endbreak.keys in ['', [], None]:  # No response was made
        exp_endbreak.keys = None
    thisExp.addData('exp_endbreak.keys',exp_endbreak.keys)
    if exp_endbreak.keys != None:  # we had a response
        thisExp.addData('exp_endbreak.rt', exp_endbreak.rt)
        thisExp.addData('exp_endbreak.duration', exp_endbreak.duration)
    thisExp.nextEntry()
    # the Routine "break_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "initRecallPics" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('initRecallPics.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from recall_c
    retTrials = 0
    retNew = 0
    retSame = 0
    random.shuffle(samePics)
    #same pics are all pictures that came from 'same' trials
    #shuffle and put in retrieval list.
    #retrievalNew = []
    retrievalSame = []
    
    retrievalSame.append(samePics)
    trial_category = "retrieval"
    # keep track of which components have finished
    initRecallPicsComponents = []
    for thisComponent in initRecallPicsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "initRecallPics" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in initRecallPicsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "initRecallPics" ---
    for thisComponent in initRecallPicsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('initRecallPics.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    # the Routine "initRecallPics" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "eyeCheck" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('eyeCheck.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    eyeCheckComponents = [driftCheck]
    for thisComponent in eyeCheckComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "eyeCheck" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.001:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in eyeCheckComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "eyeCheck" ---
    for thisComponent in eyeCheckComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('eyeCheck.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.001000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    RetrievalTrials = data.TrialHandler(nReps=numRetrieval1, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='RetrievalTrials')
    thisExp.addLoop(RetrievalTrials)  # add the loop to the experiment
    thisRetrievalTrial = RetrievalTrials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRetrievalTrial.rgb)
    if thisRetrievalTrial != None:
        for paramName in thisRetrievalTrial:
            globals()[paramName] = thisRetrievalTrial[paramName]
    
    for thisRetrievalTrial in RetrievalTrials:
        currentLoop = RetrievalTrials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisRetrievalTrial.rgb)
        if thisRetrievalTrial != None:
            for paramName in thisRetrievalTrial:
                globals()[paramName] = thisRetrievalTrial[paramName]
        
        # --- Prepare to start Routine "startRecordingRetrieval1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('startRecordingRetrieval1.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        startRecordingRetrieval1Components = [HostDrawing_Retrieval1, StartRecord_Retrieval1]
        for thisComponent in startRecordingRetrieval1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "startRecordingRetrieval1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.001:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in startRecordingRetrieval1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "startRecordingRetrieval1" ---
        for thisComponent in startRecordingRetrieval1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('startRecordingRetrieval1.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.001000)
        
        # --- Prepare to start Routine "fixation" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fixation.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        fixationComponents = [fix, MarkEvents_fixation]
        for thisComponent in fixationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fix* updates
            
            # if fix is starting this frame...
            if fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix.frameNStart = frameN  # exact frame index
                fix.tStart = t  # local t and not account for scr refresh
                fix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix.started')
                # update status
                fix.status = STARTED
                fix.setAutoDraw(True)
            
            # if fix is active this frame...
            if fix.status == STARTED:
                # update params
                pass
            
            # if fix is stopping this frame...
            if fix.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fix.tStop = t  # not accounting for scr refresh
                    fix.tStopRefresh = tThisFlipGlobal  # on global time
                    fix.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix.stopped')
                    # update status
                    fix.status = FINISHED
                    fix.setAutoDraw(False)
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('fixation.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "Q2Text" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Q2Text.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        Q2TextComponents = [question_text, MarkEvents_Q2Text]
        for thisComponent in Q2TextComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Q2Text" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > 2-frameTolerance:
                continueRoutine = False
            
            # *question_text* updates
            
            # if question_text is starting this frame...
            if question_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                question_text.frameNStart = frameN  # exact frame index
                question_text.tStart = t  # local t and not account for scr refresh
                question_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(question_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'question_text.started')
                # update status
                question_text.status = STARTED
                question_text.setAutoDraw(True)
            
            # if question_text is active this frame...
            if question_text.status == STARTED:
                # update params
                pass
            
            # if question_text is stopping this frame...
            if question_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > question_text.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    question_text.tStop = t  # not accounting for scr refresh
                    question_text.tStopRefresh = tThisFlipGlobal  # on global time
                    question_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'question_text.stopped')
                    # update status
                    question_text.status = FINISHED
                    question_text.setAutoDraw(False)
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Q2TextComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Q2Text" ---
        for thisComponent in Q2TextComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Q2Text.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "RetrievalImage" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('RetrievalImage.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from ret_img_c
        retrievalImage = None
        
        if retProb < 1.1:
            retType = "old"
            RetrievalTrials.addData("retType", "old")
            length = len(samePics)
            if(length == 0):
                sys.exit(1)
            retrievalImage = 'images/' + samePics[retSame] + '.png'
            retSame = retSame + 1
        #so I am telling it that it needs to take an image from same pics
        #and iterate through those pictures with more retrieval trials
        #no 'new' retrieval at this point.
        else:
            retType = "new"
            RetrievalTrials.addData("retType", "new")
            length = len(newPics)
            if(length == 0):
                sys.exit(1)
            retrievalImage = 'images/' + newPics[retNew] + '.png'
            retNew = retNew + 1
            
        trial_category = "retrieval"
        RetrievalTrials.addData("retIMG", retrievalImage)
        thisExp.addData("retIMG", retrievalImage)
        retIMG.setImage(retrievalImage)
        sound_pop7.setSound('POP.wav', secs=0.5, hamming=True)
        sound_pop7.setVolume(1.0, log=False)
        sound_pop7.seek(0)
        # keep track of which components have finished
        RetrievalImageComponents = [retIMG, sound_pop7, MarkEvents_Retrieval1_RetrievalImage]
        for thisComponent in RetrievalImageComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "RetrievalImage" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *retIMG* updates
            
            # if retIMG is starting this frame...
            if retIMG.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                retIMG.frameNStart = frameN  # exact frame index
                retIMG.tStart = t  # local t and not account for scr refresh
                retIMG.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(retIMG, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'retIMG.started')
                # update status
                retIMG.status = STARTED
                retIMG.setAutoDraw(True)
            
            # if retIMG is active this frame...
            if retIMG.status == STARTED:
                # update params
                pass
            
            # if retIMG is stopping this frame...
            if retIMG.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > retIMG.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    retIMG.tStop = t  # not accounting for scr refresh
                    retIMG.tStopRefresh = tThisFlipGlobal  # on global time
                    retIMG.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'retIMG.stopped')
                    # update status
                    retIMG.status = FINISHED
                    retIMG.setAutoDraw(False)
            
            # if sound_pop7 is starting this frame...
            if sound_pop7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_pop7.frameNStart = frameN  # exact frame index
                sound_pop7.tStart = t  # local t and not account for scr refresh
                sound_pop7.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_pop7.started', tThisFlipGlobal)
                # update status
                sound_pop7.status = STARTED
                sound_pop7.play(when=win)  # sync with win flip
            
            # if sound_pop7 is stopping this frame...
            if sound_pop7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_pop7.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_pop7.tStop = t  # not accounting for scr refresh
                    sound_pop7.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_pop7.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_pop7.stopped')
                    # update status
                    sound_pop7.status = FINISHED
                    sound_pop7.stop()
            # update sound_pop7 status according to whether it's playing
            if sound_pop7.isPlaying:
                sound_pop7.status = STARTED
            elif sound_pop7.isFinished:
                sound_pop7.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in RetrievalImageComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "RetrievalImage" ---
        for thisComponent in RetrievalImageComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('RetrievalImage.stopped', globalClock.getTime(format='float'))
        sound_pop7.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "Decision" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Decision.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from dec_code
        win.mouseVisible = True
        # setup some python lists for storing info about the Order_Position_Response
        Order_Position_Response.clicked_name = []
        gotValidClick = False  # until a click is received
        # keep track of which components have finished
        DecisionComponents = [decisionQ_2, first_left, second_middle, third_right, First, Second, Third, Order_Position_Response, MarkEvents_Retrieval1_Decision]
        for thisComponent in DecisionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Decision" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *decisionQ_2* updates
            
            # if decisionQ_2 is starting this frame...
            if decisionQ_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                decisionQ_2.frameNStart = frameN  # exact frame index
                decisionQ_2.tStart = t  # local t and not account for scr refresh
                decisionQ_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(decisionQ_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'decisionQ_2.started')
                # update status
                decisionQ_2.status = STARTED
                decisionQ_2.setAutoDraw(True)
            
            # if decisionQ_2 is active this frame...
            if decisionQ_2.status == STARTED:
                # update params
                pass
            
            # *first_left* updates
            
            # if first_left is starting this frame...
            if first_left.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                first_left.frameNStart = frameN  # exact frame index
                first_left.tStart = t  # local t and not account for scr refresh
                first_left.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(first_left, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'first_left.started')
                # update status
                first_left.status = STARTED
                first_left.setAutoDraw(True)
            
            # if first_left is active this frame...
            if first_left.status == STARTED:
                # update params
                pass
            
            # *second_middle* updates
            
            # if second_middle is starting this frame...
            if second_middle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                second_middle.frameNStart = frameN  # exact frame index
                second_middle.tStart = t  # local t and not account for scr refresh
                second_middle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(second_middle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'second_middle.started')
                # update status
                second_middle.status = STARTED
                second_middle.setAutoDraw(True)
            
            # if second_middle is active this frame...
            if second_middle.status == STARTED:
                # update params
                pass
            
            # *third_right* updates
            
            # if third_right is starting this frame...
            if third_right.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                third_right.frameNStart = frameN  # exact frame index
                third_right.tStart = t  # local t and not account for scr refresh
                third_right.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(third_right, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'third_right.started')
                # update status
                third_right.status = STARTED
                third_right.setAutoDraw(True)
            
            # if third_right is active this frame...
            if third_right.status == STARTED:
                # update params
                pass
            
            # *First* updates
            
            # if First is starting this frame...
            if First.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                First.frameNStart = frameN  # exact frame index
                First.tStart = t  # local t and not account for scr refresh
                First.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(First, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'First.started')
                # update status
                First.status = STARTED
                First.setAutoDraw(True)
            
            # if First is active this frame...
            if First.status == STARTED:
                # update params
                pass
            
            # *Second* updates
            
            # if Second is starting this frame...
            if Second.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Second.frameNStart = frameN  # exact frame index
                Second.tStart = t  # local t and not account for scr refresh
                Second.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Second, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Second.started')
                # update status
                Second.status = STARTED
                Second.setAutoDraw(True)
            
            # if Second is active this frame...
            if Second.status == STARTED:
                # update params
                pass
            
            # *Third* updates
            
            # if Third is starting this frame...
            if Third.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Third.frameNStart = frameN  # exact frame index
                Third.tStart = t  # local t and not account for scr refresh
                Third.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Third, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Third.started')
                # update status
                Third.status = STARTED
                Third.setAutoDraw(True)
            
            # if Third is active this frame...
            if Third.status == STARTED:
                # update params
                pass
            # *Order_Position_Response* updates
            
            # if Order_Position_Response is starting this frame...
            if Order_Position_Response.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Order_Position_Response.frameNStart = frameN  # exact frame index
                Order_Position_Response.tStart = t  # local t and not account for scr refresh
                Order_Position_Response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Order_Position_Response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('Order_Position_Response.started', t)
                # update status
                Order_Position_Response.status = STARTED
                Order_Position_Response.mouseClock.reset()
                prevButtonState = Order_Position_Response.getPressed()  # if button is down already this ISN'T a new click
            if Order_Position_Response.status == STARTED:  # only update if started and not finished!
                buttons = Order_Position_Response.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([first_left,second_middle,third_right], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(Order_Position_Response):
                                gotValidClick = True
                                Order_Position_Response.clicked_name.append(obj.name)
                        if gotValidClick:  
                            continueRoutine = False  # end routine on response
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in DecisionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Decision" ---
        for thisComponent in DecisionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Decision.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from dec_code
        retTrials = (retTrials + 1)
        RetrievalTrials.addData("retTrials", str(retTrials))
        TRIAL_INDEX = (TRIAL_INDEX + 1)
        
        RetrievalTrials.addData("TRIAL_INDEX", str(TRIAL_INDEX))
        RetrievalTrials.addData("retIMG", str(retrievalImage))
        RetrievalTrials.addData("trialCategory", str(trial_category))
        
        #for EDF output
        el_tracker.sendMessage('!V TRIAL_VAR retTrials %d' % retTrials)
        #el_tracker.sendMessage('!V TRIAL_VAR orderResponse %s' % orderResponse)
        el_tracker.sendMessage('!V TRIAL_VAR orderPosition %s' % Order_Position_Response.clicked_name)
        
        win.mouseVisible = False
        
        
        # store data for RetrievalTrials (TrialHandler)
        x, y = Order_Position_Response.getPos()
        buttons = Order_Position_Response.getPressed()
        if sum(buttons):
            # check if the mouse was inside our 'clickable' objects
            gotValidClick = False
            clickableList = environmenttools.getFromNames([first_left,second_middle,third_right], namespace=locals())
            for obj in clickableList:
                # is this object clicked on?
                if obj.contains(Order_Position_Response):
                    gotValidClick = True
                    Order_Position_Response.clicked_name.append(obj.name)
        RetrievalTrials.addData('Order_Position_Response.x', x)
        RetrievalTrials.addData('Order_Position_Response.y', y)
        RetrievalTrials.addData('Order_Position_Response.leftButton', buttons[0])
        RetrievalTrials.addData('Order_Position_Response.midButton', buttons[1])
        RetrievalTrials.addData('Order_Position_Response.rightButton', buttons[2])
        if len(Order_Position_Response.clicked_name):
            RetrievalTrials.addData('Order_Position_Response.clicked_name', Order_Position_Response.clicked_name[0])
        # the Routine "Decision" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "stopRecording" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('stopRecording.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        stopRecordingComponents = [StopRecord]
        for thisComponent in stopRecordingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "stopRecording" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.001:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in stopRecordingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "stopRecording" ---
        for thisComponent in stopRecordingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('stopRecording.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.001000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed numRetrieval1 repeats of 'RetrievalTrials'
    
    
    # --- Prepare to start Routine "init2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('init2.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from init2code
    sameDiff2 = ["same","same","same","same","same","same","same","same","same","same","same","same","same","same","same","same","same","same","same","same","diff","diff","diff","diff","diff"];
    shuffle(sameDiff2)
    
    #So for the same trials:
    # Define the strings
    fixed_strings = ['LMR', 'RML']  # These will occur 7 times each
    other_strings = ['MLR', 'MRL', 'LRM', 'RLM']  # These will be randomly selected to have total of 7 also
    
    # Select 7 strings from other_strings with repetitions
    selected_strings = random.choices(other_strings, k=9)
    
    # Create a list to store the strings
    sameTrials2 = fixed_strings * 9  # Repeating fixed_strings 9 times
    sameTrials2 += selected_strings  # Adding the selected 9 strings
    
    # Shuffle the list
    random.shuffle(sameTrials2)
    
    # Print the randomized list
    print("Randomized string list:")
    print(sameTrials) #now this should be a list of 27 that we will run 20 of, but because it is indexed by eTrials, there needs to be more than 25 items. 
    
    #for the different trials:
    diffTrials2 = fixed_strings * 9  # Repeating fixed_strings 9 times
    diffTrials2 += selected_strings  # Adding the selected 9 strings
    
    # Shuffle the list
    random.shuffle(diffTrials2)
    
    # Print the selected strings)
    print(diffTrials2) #because these are indexed by eTrials while they select the images, they need to be more than 25 items
    
    #hard-coded names of all images and trial types (counterbalanced).
    numItems = 210
    #numTotencoding = 50
    numEncodingBlock1 = 25
    numEncodingBlock2 = 25
    numTotretrieval = 40
    numRetrieval1 = 20
    numRetrieval2 = 20
    eTrials2 = 0
    
    #define again
    newPics2 = []
    samePics2 = [] 
    # keep track of which components have finished
    init2Components = []
    for thisComponent in init2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "init2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in init2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "init2" ---
    for thisComponent in init2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('init2.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    # the Routine "init2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "trial" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('trial.started', globalClock.getTime(format='float'))
    start.keys = []
    start.rt = []
    _start_allKeys = []
    # keep track of which components have finished
    trialComponents = [Instructions, start]
    for thisComponent in trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "trial" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Instructions* updates
        
        # if Instructions is starting this frame...
        if Instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Instructions.frameNStart = frameN  # exact frame index
            Instructions.tStart = t  # local t and not account for scr refresh
            Instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Instructions.started')
            # update status
            Instructions.status = STARTED
            Instructions.setAutoDraw(True)
        
        # if Instructions is active this frame...
        if Instructions.status == STARTED:
            # update params
            pass
        
        # *start* updates
        waitOnFlip = False
        
        # if start is starting this frame...
        if start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start.frameNStart = frameN  # exact frame index
            start.tStart = t  # local t and not account for scr refresh
            start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start.started')
            # update status
            start.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start.clock.reset)  # t=0 on next screen flip
        if start.status == STARTED and not waitOnFlip:
            theseKeys = start.getKeys(keyList=['space','q'], ignoreKeys=None, waitRelease=False)
            _start_allKeys.extend(theseKeys)
            if len(_start_allKeys):
                start.keys = _start_allKeys[-1].name  # just the last key pressed
                start.rt = _start_allKeys[-1].rt
                start.duration = _start_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "trial" ---
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('trial.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    # the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    EncodingBlock2 = data.TrialHandler(nReps=numEncodingBlock2, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='EncodingBlock2')
    thisExp.addLoop(EncodingBlock2)  # add the loop to the experiment
    thisEncodingBlock2 = EncodingBlock2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisEncodingBlock2.rgb)
    if thisEncodingBlock2 != None:
        for paramName in thisEncodingBlock2:
            globals()[paramName] = thisEncodingBlock2[paramName]
    
    for thisEncodingBlock2 in EncodingBlock2:
        currentLoop = EncodingBlock2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisEncodingBlock2.rgb)
        if thisEncodingBlock2 != None:
            for paramName in thisEncodingBlock2:
                globals()[paramName] = thisEncodingBlock2[paramName]
        
        # --- Prepare to start Routine "startRecordingEncoding2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('startRecordingEncoding2.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        startRecordingEncoding2Components = [HostDrawing_Encoding2, StartRecord_Encoding2]
        for thisComponent in startRecordingEncoding2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "startRecordingEncoding2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.001:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in startRecordingEncoding2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "startRecordingEncoding2" ---
        for thisComponent in startRecordingEncoding2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('startRecordingEncoding2.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.001000)
        
        # --- Prepare to start Routine "fixation" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fixation.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        fixationComponents = [fix, MarkEvents_fixation]
        for thisComponent in fixationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fix* updates
            
            # if fix is starting this frame...
            if fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix.frameNStart = frameN  # exact frame index
                fix.tStart = t  # local t and not account for scr refresh
                fix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix.started')
                # update status
                fix.status = STARTED
                fix.setAutoDraw(True)
            
            # if fix is active this frame...
            if fix.status == STARTED:
                # update params
                pass
            
            # if fix is stopping this frame...
            if fix.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fix.tStop = t  # not accounting for scr refresh
                    fix.tStopRefresh = tThisFlipGlobal  # on global time
                    fix.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix.stopped')
                    # update status
                    fix.status = FINISHED
                    fix.setAutoDraw(False)
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('fixation.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "first_view2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('first_view2.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from firstw_code_2
        
        #iterate through trial type and set a new one for each trial
        #trialType = trialTypes[eTrials]
        #method_Trial = sameDiff[eTrials] 
        
        #iterate through trial type and set a new one for each trial
        method_Trial = sameDiff2[eTrials2]
        if method_Trial == "same":
            trialType = sameTrials2[eTrials2]
        else:
            trialType = diffTrials2[eTrials]
            #positions
        leftPos = -0.5, 0.0
        midPos = 0.0, 0.0
        rightPos = 0.5, 0.0
            
        if trialType == 'LMR':
                firstPos = leftPos
                secondPos = midPos
                thirdPos = rightPos
        elif trialType == 'LRM':
                firstPos = leftPos
                secondPos = rightPos
                thirdPos = midPos
        elif trialType == 'MLR':
                firstPos = midPos
                secondPos = leftPos
                thirdPos = rightPos
        elif trialType == 'RML':
                firstPos = rightPos
                secondPos = midPos
                thirdPos = leftPos
        elif trialType == 'RLM':
                firstPos = rightPos
                secondPos = leftPos
                thirdPos = midPos
        elif trialType == 'MRL':
                firstPos = midPos
                secondPos = rightPos
                thirdPos = leftPos
        #set images in triplets
        triplet1 = 'images/'+encodingPics2[eTrials2][0]+'.png'
        triplet2 = 'images/'+encodingPics2[eTrials2][1]+'.png'
        triplet3 = 'images/'+encodingPics2[eTrials2][2]+'.png'
        
        firstPos1 = firstPos
        secondPos1 = secondPos
        thirdPos1 = thirdPos
        triplet1_1 = triplet1
        triplet2_1 = triplet2
        triplet3_1 = triplet3 
        
        
        EncodingBlock2.addData ("Trial_Order", trialType)
         #   thisExp.addData("TrialOrder", trialType)
        thisExp.addData("triplet1", triplet1)
        thisExp.addData("triplet2", triplet2)
        thisExp.addData("triplet3", triplet3)
        trial_category = "encoding"
        Triplet1_2.setPos(firstPos)
        Triplet1_2.setImage(triplet1)
        Triplet2_2.setPos(secondPos)
        Triplet2_2.setImage(triplet2)
        Triplet3_2.setPos(thirdPos)
        Triplet3_2.setImage(triplet3)
        sound_pop_2.setSound('POP.wav', secs=1.5, hamming=True)
        sound_pop_2.setVolume(1.0, log=False)
        sound_pop_2.seek(0)
        sound_pop2_2.setSound('POP.wav', secs=1.5, hamming=True)
        sound_pop2_2.setVolume(1.0, log=False)
        sound_pop2_2.seek(0)
        sound_pop3_2.setSound('POP.wav', secs=1.5, hamming=True)
        sound_pop3_2.setVolume(1.0, log=False)
        sound_pop3_2.seek(0)
        # keep track of which components have finished
        first_view2Components = [Triplet1_2, Triplet2_2, Triplet3_2, sound_pop_2, sound_pop2_2, sound_pop3_2, MarkEvents_Encoding2_first_view2]
        for thisComponent in first_view2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "first_view2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Triplet1_2* updates
            
            # if Triplet1_2 is starting this frame...
            if Triplet1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Triplet1_2.frameNStart = frameN  # exact frame index
                Triplet1_2.tStart = t  # local t and not account for scr refresh
                Triplet1_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Triplet1_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Triplet1_2.started')
                # update status
                Triplet1_2.status = STARTED
                Triplet1_2.setAutoDraw(True)
            
            # if Triplet1_2 is active this frame...
            if Triplet1_2.status == STARTED:
                # update params
                pass
            
            # if Triplet1_2 is stopping this frame...
            if Triplet1_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Triplet1_2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Triplet1_2.tStop = t  # not accounting for scr refresh
                    Triplet1_2.tStopRefresh = tThisFlipGlobal  # on global time
                    Triplet1_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Triplet1_2.stopped')
                    # update status
                    Triplet1_2.status = FINISHED
                    Triplet1_2.setAutoDraw(False)
            
            # *Triplet2_2* updates
            
            # if Triplet2_2 is starting this frame...
            if Triplet2_2.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                Triplet2_2.frameNStart = frameN  # exact frame index
                Triplet2_2.tStart = t  # local t and not account for scr refresh
                Triplet2_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Triplet2_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Triplet2_2.started')
                # update status
                Triplet2_2.status = STARTED
                Triplet2_2.setAutoDraw(True)
            
            # if Triplet2_2 is active this frame...
            if Triplet2_2.status == STARTED:
                # update params
                pass
            
            # if Triplet2_2 is stopping this frame...
            if Triplet2_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Triplet2_2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Triplet2_2.tStop = t  # not accounting for scr refresh
                    Triplet2_2.tStopRefresh = tThisFlipGlobal  # on global time
                    Triplet2_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Triplet2_2.stopped')
                    # update status
                    Triplet2_2.status = FINISHED
                    Triplet2_2.setAutoDraw(False)
            
            # *Triplet3_2* updates
            
            # if Triplet3_2 is starting this frame...
            if Triplet3_2.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                Triplet3_2.frameNStart = frameN  # exact frame index
                Triplet3_2.tStart = t  # local t and not account for scr refresh
                Triplet3_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Triplet3_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Triplet3_2.started')
                # update status
                Triplet3_2.status = STARTED
                Triplet3_2.setAutoDraw(True)
            
            # if Triplet3_2 is active this frame...
            if Triplet3_2.status == STARTED:
                # update params
                pass
            
            # if Triplet3_2 is stopping this frame...
            if Triplet3_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Triplet3_2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Triplet3_2.tStop = t  # not accounting for scr refresh
                    Triplet3_2.tStopRefresh = tThisFlipGlobal  # on global time
                    Triplet3_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Triplet3_2.stopped')
                    # update status
                    Triplet3_2.status = FINISHED
                    Triplet3_2.setAutoDraw(False)
            
            # if sound_pop_2 is starting this frame...
            if sound_pop_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                sound_pop_2.frameNStart = frameN  # exact frame index
                sound_pop_2.tStart = t  # local t and not account for scr refresh
                sound_pop_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_pop_2.started', tThisFlipGlobal)
                # update status
                sound_pop_2.status = STARTED
                sound_pop_2.play(when=win)  # sync with win flip
            
            # if sound_pop_2 is stopping this frame...
            if sound_pop_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_pop_2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_pop_2.tStop = t  # not accounting for scr refresh
                    sound_pop_2.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_pop_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_pop_2.stopped')
                    # update status
                    sound_pop_2.status = FINISHED
                    sound_pop_2.stop()
            # update sound_pop_2 status according to whether it's playing
            if sound_pop_2.isPlaying:
                sound_pop_2.status = STARTED
            elif sound_pop_2.isFinished:
                sound_pop_2.status = FINISHED
            
            # if sound_pop2_2 is starting this frame...
            if sound_pop2_2.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                sound_pop2_2.frameNStart = frameN  # exact frame index
                sound_pop2_2.tStart = t  # local t and not account for scr refresh
                sound_pop2_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_pop2_2.started', tThisFlipGlobal)
                # update status
                sound_pop2_2.status = STARTED
                sound_pop2_2.play(when=win)  # sync with win flip
            
            # if sound_pop2_2 is stopping this frame...
            if sound_pop2_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_pop2_2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_pop2_2.tStop = t  # not accounting for scr refresh
                    sound_pop2_2.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_pop2_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_pop2_2.stopped')
                    # update status
                    sound_pop2_2.status = FINISHED
                    sound_pop2_2.stop()
            # update sound_pop2_2 status according to whether it's playing
            if sound_pop2_2.isPlaying:
                sound_pop2_2.status = STARTED
            elif sound_pop2_2.isFinished:
                sound_pop2_2.status = FINISHED
            
            # if sound_pop3_2 is starting this frame...
            if sound_pop3_2.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                sound_pop3_2.frameNStart = frameN  # exact frame index
                sound_pop3_2.tStart = t  # local t and not account for scr refresh
                sound_pop3_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_pop3_2.started', tThisFlipGlobal)
                # update status
                sound_pop3_2.status = STARTED
                sound_pop3_2.play(when=win)  # sync with win flip
            
            # if sound_pop3_2 is stopping this frame...
            if sound_pop3_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_pop3_2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_pop3_2.tStop = t  # not accounting for scr refresh
                    sound_pop3_2.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_pop3_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_pop3_2.stopped')
                    # update status
                    sound_pop3_2.status = FINISHED
                    sound_pop3_2.stop()
            # update sound_pop3_2 status according to whether it's playing
            if sound_pop3_2.isPlaying:
                sound_pop3_2.status = STARTED
            elif sound_pop3_2.isFinished:
                sound_pop3_2.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in first_view2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "first_view2" ---
        for thisComponent in first_view2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('first_view2.stopped', globalClock.getTime(format='float'))
        sound_pop_2.pause()  # ensure sound has stopped at end of Routine
        sound_pop2_2.pause()  # ensure sound has stopped at end of Routine
        sound_pop3_2.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.500000)
        
        # --- Prepare to start Routine "mask" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('mask.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        maskComponents = [Mask, MarkEvents_mask]
        for thisComponent in maskComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "mask" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Mask* updates
            
            # if Mask is starting this frame...
            if Mask.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Mask.frameNStart = frameN  # exact frame index
                Mask.tStart = t  # local t and not account for scr refresh
                Mask.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Mask, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Mask.started')
                # update status
                Mask.status = STARTED
                Mask.setAutoDraw(True)
            
            # if Mask is active this frame...
            if Mask.status == STARTED:
                # update params
                pass
            
            # if Mask is stopping this frame...
            if Mask.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Mask.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    Mask.tStop = t  # not accounting for scr refresh
                    Mask.tStopRefresh = tThisFlipGlobal  # on global time
                    Mask.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Mask.stopped')
                    # update status
                    Mask.status = FINISHED
                    Mask.setAutoDraw(False)
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in maskComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "mask" ---
        for thisComponent in maskComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('mask.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "second_view2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('second_view2.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from secondw_code_2
         #used to be same or different trials based on random number, but now making sure there are 20 same trials and 5 diff trials
        if method_Trial == "same":
            list = [0,1,2]
            random.shuffle(list)
            samePics2.append(encodingPics2[eTrials2][list[0]])
            EncodingBlock2.addData("encodingType","same")
            encodingType = 'same'
            methodAns = ['leftCL_2'] #same
        else:
            nextTrial = random.choice(diffTrials2)
            while nextTrial == trialType:
                nextTrial = random.choice(diffTrials2)
                end
            trialType = nextTrial
            methodAns = ['rightCL_2'] #different
            list = [0,1,2]
            random.shuffle(list)
            triplet1 = 'images/'+encodingPics2[eTrials2][list[0]]+'.png'
            triplet2 = 'images/'+encodingPics2[eTrials2][list[1]]+'.png'
            triplet3 = 'images/'+encodingPics2[eTrials2][list[2]]+'.png'
            EncodingBlock2.addData("encodingType","diff")
            encodingType = 'diff'
        
        #have to re-state positions and trial types
        if trialType == 'LMR':
            firstPos = leftPos
            secondPos = midPos
            thirdPos = rightPos
        elif trialType == 'LRM':
            firstPos = leftPos
            secondPos = rightPos
            thirdPos = midPos
        elif trialType == 'MLR':
            firstPos = midPos
            secondPos = leftPos
            thirdPos = rightPos
        elif trialType == 'RML':
            firstPos = rightPos
            secondPos = midPos
            thirdPos = leftPos
        elif trialType == 'RLM':
            firstPos = rightPos
            secondPos = leftPos
            thirdPos = midPos
        elif trialType == 'MRL':
            firstPos = midPos
            secondPos = rightPos
            thirdPos = leftPos
            
        firstPos2 = firstPos
        secondPos2 = secondPos
        thirdPos2 = thirdPos
        triplet1_2 = triplet1
        triplet2_2 = triplet2
        triplet3_2 = triplet3     
        
        EncodingBlock2.addData("triplet1", triplet1)
        EncodingBlock2.addData("triplet2", triplet2)
        EncodingBlock2.addData("triplet3", triplet3)
        EncodingBlock2.addData("nextTrial", nextTrial)
        EncodingBlock2.addData("Trial_Order", trialType)
        img1_2.setPos(firstPos)
        img1_2.setImage(triplet1)
        img2_2.setPos(secondPos)
        img2_2.setImage(triplet2)
        img3_2.setPos(thirdPos)
        img3_2.setImage(triplet3)
        sound_pop4_2.setSound('POP.wav', secs=1.5, hamming=True)
        sound_pop4_2.setVolume(1.0, log=False)
        sound_pop4_2.seek(0)
        sound_pop5_2.setSound('POP.wav', secs=1.5, hamming=True)
        sound_pop5_2.setVolume(1.0, log=False)
        sound_pop5_2.seek(0)
        sound_pop6_2.setSound('POP.wav', secs=1.5, hamming=True)
        sound_pop6_2.setVolume(1.0, log=False)
        sound_pop6_2.seek(0)
        # keep track of which components have finished
        second_view2Components = [img1_2, img2_2, img3_2, sound_pop4_2, sound_pop5_2, sound_pop6_2, MarkEvents_Encoding2_second_view2]
        for thisComponent in second_view2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "second_view2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *img1_2* updates
            
            # if img1_2 is starting this frame...
            if img1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                img1_2.frameNStart = frameN  # exact frame index
                img1_2.tStart = t  # local t and not account for scr refresh
                img1_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img1_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img1_2.started')
                # update status
                img1_2.status = STARTED
                img1_2.setAutoDraw(True)
            
            # if img1_2 is active this frame...
            if img1_2.status == STARTED:
                # update params
                pass
            
            # if img1_2 is stopping this frame...
            if img1_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > img1_2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    img1_2.tStop = t  # not accounting for scr refresh
                    img1_2.tStopRefresh = tThisFlipGlobal  # on global time
                    img1_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'img1_2.stopped')
                    # update status
                    img1_2.status = FINISHED
                    img1_2.setAutoDraw(False)
            
            # *img2_2* updates
            
            # if img2_2 is starting this frame...
            if img2_2.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                img2_2.frameNStart = frameN  # exact frame index
                img2_2.tStart = t  # local t and not account for scr refresh
                img2_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img2_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img2_2.started')
                # update status
                img2_2.status = STARTED
                img2_2.setAutoDraw(True)
            
            # if img2_2 is active this frame...
            if img2_2.status == STARTED:
                # update params
                pass
            
            # if img2_2 is stopping this frame...
            if img2_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > img2_2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    img2_2.tStop = t  # not accounting for scr refresh
                    img2_2.tStopRefresh = tThisFlipGlobal  # on global time
                    img2_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'img2_2.stopped')
                    # update status
                    img2_2.status = FINISHED
                    img2_2.setAutoDraw(False)
            
            # *img3_2* updates
            
            # if img3_2 is starting this frame...
            if img3_2.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                img3_2.frameNStart = frameN  # exact frame index
                img3_2.tStart = t  # local t and not account for scr refresh
                img3_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img3_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img3_2.started')
                # update status
                img3_2.status = STARTED
                img3_2.setAutoDraw(True)
            
            # if img3_2 is active this frame...
            if img3_2.status == STARTED:
                # update params
                pass
            
            # if img3_2 is stopping this frame...
            if img3_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > img3_2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    img3_2.tStop = t  # not accounting for scr refresh
                    img3_2.tStopRefresh = tThisFlipGlobal  # on global time
                    img3_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'img3_2.stopped')
                    # update status
                    img3_2.status = FINISHED
                    img3_2.setAutoDraw(False)
            
            # if sound_pop4_2 is starting this frame...
            if sound_pop4_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_pop4_2.frameNStart = frameN  # exact frame index
                sound_pop4_2.tStart = t  # local t and not account for scr refresh
                sound_pop4_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_pop4_2.started', tThisFlipGlobal)
                # update status
                sound_pop4_2.status = STARTED
                sound_pop4_2.play(when=win)  # sync with win flip
            
            # if sound_pop4_2 is stopping this frame...
            if sound_pop4_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_pop4_2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_pop4_2.tStop = t  # not accounting for scr refresh
                    sound_pop4_2.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_pop4_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_pop4_2.stopped')
                    # update status
                    sound_pop4_2.status = FINISHED
                    sound_pop4_2.stop()
            # update sound_pop4_2 status according to whether it's playing
            if sound_pop4_2.isPlaying:
                sound_pop4_2.status = STARTED
            elif sound_pop4_2.isFinished:
                sound_pop4_2.status = FINISHED
            
            # if sound_pop5_2 is starting this frame...
            if sound_pop5_2.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                sound_pop5_2.frameNStart = frameN  # exact frame index
                sound_pop5_2.tStart = t  # local t and not account for scr refresh
                sound_pop5_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_pop5_2.started', tThisFlipGlobal)
                # update status
                sound_pop5_2.status = STARTED
                sound_pop5_2.play(when=win)  # sync with win flip
            
            # if sound_pop5_2 is stopping this frame...
            if sound_pop5_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_pop5_2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_pop5_2.tStop = t  # not accounting for scr refresh
                    sound_pop5_2.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_pop5_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_pop5_2.stopped')
                    # update status
                    sound_pop5_2.status = FINISHED
                    sound_pop5_2.stop()
            # update sound_pop5_2 status according to whether it's playing
            if sound_pop5_2.isPlaying:
                sound_pop5_2.status = STARTED
            elif sound_pop5_2.isFinished:
                sound_pop5_2.status = FINISHED
            
            # if sound_pop6_2 is starting this frame...
            if sound_pop6_2.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                sound_pop6_2.frameNStart = frameN  # exact frame index
                sound_pop6_2.tStart = t  # local t and not account for scr refresh
                sound_pop6_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_pop6_2.started', tThisFlipGlobal)
                # update status
                sound_pop6_2.status = STARTED
                sound_pop6_2.play(when=win)  # sync with win flip
            
            # if sound_pop6_2 is stopping this frame...
            if sound_pop6_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_pop6_2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_pop6_2.tStop = t  # not accounting for scr refresh
                    sound_pop6_2.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_pop6_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_pop6_2.stopped')
                    # update status
                    sound_pop6_2.status = FINISHED
                    sound_pop6_2.stop()
            # update sound_pop6_2 status according to whether it's playing
            if sound_pop6_2.isPlaying:
                sound_pop6_2.status = STARTED
            elif sound_pop6_2.isFinished:
                sound_pop6_2.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in second_view2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "second_view2" ---
        for thisComponent in second_view2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('second_view2.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from secondw_code_2
        eTrials2 = eTrials2 + 1
        TRIAL_INDEX = TRIAL_INDEX + 1
        
        #for psychopy
        EncodingBlock2.addData("encodingTrials", str(eTrials2))
        EncodingBlock2.addData("TRIAL_INDEX", str(TRIAL_INDEX))   
        
        #for EDF file
        el_tracker.sendMessage('!V TRIAL_VAR TRIAL_INDEX %d' % TRIAL_INDEX)
        el_tracker.sendMessage('!V TRIAL_VAR eTrials %d' %eTrials2)
        sound_pop4_2.pause()  # ensure sound has stopped at end of Routine
        sound_pop5_2.pause()  # ensure sound has stopped at end of Routine
        sound_pop6_2.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.500000)
        
        # --- Prepare to start Routine "methodQText" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('methodQText.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        methodQTextComponents = [text_2, MarkEvents_methodQText]
        for thisComponent in methodQTextComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "methodQText" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.started')
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # if text_2 is stopping this frame...
            if text_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_2.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_2.tStop = t  # not accounting for scr refresh
                    text_2.tStopRefresh = tThisFlipGlobal  # on global time
                    text_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_2.stopped')
                    # update status
                    text_2.status = FINISHED
                    text_2.setAutoDraw(False)
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in methodQTextComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "methodQText" ---
        for thisComponent in methodQTextComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('methodQText.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "MethodCheck2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('MethodCheck2.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from methodc_code_2
        win.mouseVisible = True
        sameText_2.setText('Same')
        diffText_2.setText('Different')
        # setup some python lists for storing info about the MethodMouse_2
        MethodMouse_2.x = []
        MethodMouse_2.y = []
        MethodMouse_2.leftButton = []
        MethodMouse_2.midButton = []
        MethodMouse_2.rightButton = []
        MethodMouse_2.time = []
        MethodMouse_2.clicked_name = []
        gotValidClick = False  # until a click is received
        # keep track of which components have finished
        MethodCheck2Components = [text_5, sameText_2, diffText_2, leftCL_2, rightCL_2, MethodMouse_2, MarkEvents_Encoding2_MethodCheck2]
        for thisComponent in MethodCheck2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "MethodCheck2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_5* updates
            
            # if text_5 is starting this frame...
            if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_5.frameNStart = frameN  # exact frame index
                text_5.tStart = t  # local t and not account for scr refresh
                text_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_5.started')
                # update status
                text_5.status = STARTED
                text_5.setAutoDraw(True)
            
            # if text_5 is active this frame...
            if text_5.status == STARTED:
                # update params
                pass
            
            # *sameText_2* updates
            
            # if sameText_2 is starting this frame...
            if sameText_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sameText_2.frameNStart = frameN  # exact frame index
                sameText_2.tStart = t  # local t and not account for scr refresh
                sameText_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sameText_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sameText_2.started')
                # update status
                sameText_2.status = STARTED
                sameText_2.setAutoDraw(True)
            
            # if sameText_2 is active this frame...
            if sameText_2.status == STARTED:
                # update params
                pass
            
            # *diffText_2* updates
            
            # if diffText_2 is starting this frame...
            if diffText_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                diffText_2.frameNStart = frameN  # exact frame index
                diffText_2.tStart = t  # local t and not account for scr refresh
                diffText_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(diffText_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'diffText_2.started')
                # update status
                diffText_2.status = STARTED
                diffText_2.setAutoDraw(True)
            
            # if diffText_2 is active this frame...
            if diffText_2.status == STARTED:
                # update params
                pass
            
            # *leftCL_2* updates
            
            # if leftCL_2 is starting this frame...
            if leftCL_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                leftCL_2.frameNStart = frameN  # exact frame index
                leftCL_2.tStart = t  # local t and not account for scr refresh
                leftCL_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(leftCL_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'leftCL_2.started')
                # update status
                leftCL_2.status = STARTED
                leftCL_2.setAutoDraw(True)
            
            # if leftCL_2 is active this frame...
            if leftCL_2.status == STARTED:
                # update params
                pass
            
            # *rightCL_2* updates
            
            # if rightCL_2 is starting this frame...
            if rightCL_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rightCL_2.frameNStart = frameN  # exact frame index
                rightCL_2.tStart = t  # local t and not account for scr refresh
                rightCL_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rightCL_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rightCL_2.started')
                # update status
                rightCL_2.status = STARTED
                rightCL_2.setAutoDraw(True)
            
            # if rightCL_2 is active this frame...
            if rightCL_2.status == STARTED:
                # update params
                pass
            # *MethodMouse_2* updates
            
            # if MethodMouse_2 is starting this frame...
            if MethodMouse_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                MethodMouse_2.frameNStart = frameN  # exact frame index
                MethodMouse_2.tStart = t  # local t and not account for scr refresh
                MethodMouse_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(MethodMouse_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('MethodMouse_2.started', t)
                # update status
                MethodMouse_2.status = STARTED
                MethodMouse_2.mouseClock.reset()
                prevButtonState = MethodMouse_2.getPressed()  # if button is down already this ISN'T a new click
            if MethodMouse_2.status == STARTED:  # only update if started and not finished!
                buttons = MethodMouse_2.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([rightCL_2, leftCL_2], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(MethodMouse_2):
                                gotValidClick = True
                                MethodMouse_2.clicked_name.append(obj.name)
                        x, y = MethodMouse_2.getPos()
                        MethodMouse_2.x.append(x)
                        MethodMouse_2.y.append(y)
                        buttons = MethodMouse_2.getPressed()
                        MethodMouse_2.leftButton.append(buttons[0])
                        MethodMouse_2.midButton.append(buttons[1])
                        MethodMouse_2.rightButton.append(buttons[2])
                        MethodMouse_2.time.append(MethodMouse_2.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in MethodCheck2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "MethodCheck2" ---
        for thisComponent in MethodCheck2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('MethodCheck2.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from methodc_code_2
        #store whether or not they answered correctly for the "same" or "different" trial type.
        if MethodMouse_2.clicked_name == methodAns:
            EncodingBlock2.addData ("methodCheck", '1')
            methodCheck = 1
        else:
            EncodingBlock2.addData ("methodCheck", '0')
            methodCheck = 0
        
        #psychopy output
        EncodingBlock2.addData("TRIAL_INDEX", str(TRIAL_INDEX))
        EncodingBlock2.addData("trialCategory", str(trial_category))
        EncodingBlock2.addData("encodingTrials", str(eTrials2))
        EncodingBlock2.addData("methodCheck", str(methodCheck))
        
        #for the EDF output
        el_tracker.sendMessage('!V TRIAL_VAR methodCheck %s' % methodCheck)
        el_tracker.sendMessage('!V TRIAL_VAR methodResp %s' % MethodMouse_2.clicked_name)
        
        win.mouseVisible = False
        # store data for EncodingBlock2 (TrialHandler)
        EncodingBlock2.addData('MethodMouse_2.x', MethodMouse_2.x)
        EncodingBlock2.addData('MethodMouse_2.y', MethodMouse_2.y)
        EncodingBlock2.addData('MethodMouse_2.leftButton', MethodMouse_2.leftButton)
        EncodingBlock2.addData('MethodMouse_2.midButton', MethodMouse_2.midButton)
        EncodingBlock2.addData('MethodMouse_2.rightButton', MethodMouse_2.rightButton)
        EncodingBlock2.addData('MethodMouse_2.time', MethodMouse_2.time)
        EncodingBlock2.addData('MethodMouse_2.clicked_name', MethodMouse_2.clicked_name)
        # the Routine "MethodCheck2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "stopRecording" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('stopRecording.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        stopRecordingComponents = [StopRecord]
        for thisComponent in stopRecordingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "stopRecording" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.001:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in stopRecordingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "stopRecording" ---
        for thisComponent in stopRecordingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('stopRecording.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.001000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed numEncodingBlock2 repeats of 'EncodingBlock2'
    
    
    # --- Prepare to start Routine "break_1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('break_1.started', globalClock.getTime(format='float'))
    start2.keys = []
    start2.rt = []
    _start2_allKeys = []
    exp_endbreak.keys = []
    exp_endbreak.rt = []
    _exp_endbreak_allKeys = []
    # keep track of which components have finished
    break_1Components = [BlockBreak, text_4, start2, exp_endbreak]
    for thisComponent in break_1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *BlockBreak* updates
        
        # if BlockBreak is starting this frame...
        if BlockBreak.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            BlockBreak.frameNStart = frameN  # exact frame index
            BlockBreak.tStart = t  # local t and not account for scr refresh
            BlockBreak.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(BlockBreak, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'BlockBreak.started')
            # update status
            BlockBreak.status = STARTED
            BlockBreak.setAutoDraw(True)
        
        # if BlockBreak is active this frame...
        if BlockBreak.status == STARTED:
            # update params
            pass
        
        # if BlockBreak is stopping this frame...
        if BlockBreak.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > BlockBreak.tStartRefresh + 180-frameTolerance:
                # keep track of stop time/frame for later
                BlockBreak.tStop = t  # not accounting for scr refresh
                BlockBreak.tStopRefresh = tThisFlipGlobal  # on global time
                BlockBreak.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'BlockBreak.stopped')
                # update status
                BlockBreak.status = FINISHED
                BlockBreak.setAutoDraw(False)
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 180.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_4.started')
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # *start2* updates
        waitOnFlip = False
        
        # if start2 is starting this frame...
        if start2.status == NOT_STARTED and tThisFlip >= 180.0-frameTolerance:
            # keep track of start time/frame for later
            start2.frameNStart = frameN  # exact frame index
            start2.tStart = t  # local t and not account for scr refresh
            start2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start2.started')
            # update status
            start2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(start2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if start2.status == STARTED and not waitOnFlip:
            theseKeys = start2.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _start2_allKeys.extend(theseKeys)
            if len(_start2_allKeys):
                start2.keys = _start2_allKeys[-1].name  # just the last key pressed
                start2.rt = _start2_allKeys[-1].rt
                start2.duration = _start2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *exp_endbreak* updates
        waitOnFlip = False
        
        # if exp_endbreak is starting this frame...
        if exp_endbreak.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            exp_endbreak.frameNStart = frameN  # exact frame index
            exp_endbreak.tStart = t  # local t and not account for scr refresh
            exp_endbreak.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(exp_endbreak, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'exp_endbreak.started')
            # update status
            exp_endbreak.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(exp_endbreak.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(exp_endbreak.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if exp_endbreak.status == STARTED and not waitOnFlip:
            theseKeys = exp_endbreak.getKeys(keyList=['z'], ignoreKeys=None, waitRelease=False)
            _exp_endbreak_allKeys.extend(theseKeys)
            if len(_exp_endbreak_allKeys):
                exp_endbreak.keys = _exp_endbreak_allKeys[-1].name  # just the last key pressed
                exp_endbreak.rt = _exp_endbreak_allKeys[-1].rt
                exp_endbreak.duration = _exp_endbreak_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_1" ---
    for thisComponent in break_1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('break_1.stopped', globalClock.getTime(format='float'))
    # check responses
    if start2.keys in ['', [], None]:  # No response was made
        start2.keys = None
    thisExp.addData('start2.keys',start2.keys)
    if start2.keys != None:  # we had a response
        thisExp.addData('start2.rt', start2.rt)
        thisExp.addData('start2.duration', start2.duration)
    # check responses
    if exp_endbreak.keys in ['', [], None]:  # No response was made
        exp_endbreak.keys = None
    thisExp.addData('exp_endbreak.keys',exp_endbreak.keys)
    if exp_endbreak.keys != None:  # we had a response
        thisExp.addData('exp_endbreak.rt', exp_endbreak.rt)
        thisExp.addData('exp_endbreak.duration', exp_endbreak.duration)
    thisExp.nextEntry()
    # the Routine "break_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "initRecallPics2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('initRecallPics2.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from recall_c_2
    retTrials2 = 0
    retNew2 = 0
    retSame2 = 0
    random.shuffle(samePics2)
    #same pics are all pictures that came from 'same' trials
    #shuffle and put in retrieval list.
    #retrievalNew = []
    retrievalSame2 = []
    
    retrievalSame2.append(samePics2)
    trial_category = "retrieval"
    print("made it to the beginning of the recall routine")
    # keep track of which components have finished
    initRecallPics2Components = []
    for thisComponent in initRecallPics2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "initRecallPics2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in initRecallPics2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "initRecallPics2" ---
    for thisComponent in initRecallPics2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('initRecallPics2.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    # the Routine "initRecallPics2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "eyeCheck2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('eyeCheck2.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    eyeCheck2Components = [drift2]
    for thisComponent in eyeCheck2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "eyeCheck2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.001:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in eyeCheck2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "eyeCheck2" ---
    for thisComponent in eyeCheck2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('eyeCheck2.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.001000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    RetrievalBlock2 = data.TrialHandler(nReps=numRetrieval2, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='RetrievalBlock2')
    thisExp.addLoop(RetrievalBlock2)  # add the loop to the experiment
    thisRetrievalBlock2 = RetrievalBlock2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRetrievalBlock2.rgb)
    if thisRetrievalBlock2 != None:
        for paramName in thisRetrievalBlock2:
            globals()[paramName] = thisRetrievalBlock2[paramName]
    
    for thisRetrievalBlock2 in RetrievalBlock2:
        currentLoop = RetrievalBlock2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisRetrievalBlock2.rgb)
        if thisRetrievalBlock2 != None:
            for paramName in thisRetrievalBlock2:
                globals()[paramName] = thisRetrievalBlock2[paramName]
        
        # --- Prepare to start Routine "startRecording_Retrieval2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('startRecording_Retrieval2.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        startRecording_Retrieval2Components = [HostDrawing_Retrieval2, StartRecord_Retrieval2]
        for thisComponent in startRecording_Retrieval2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "startRecording_Retrieval2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.001:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in startRecording_Retrieval2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "startRecording_Retrieval2" ---
        for thisComponent in startRecording_Retrieval2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('startRecording_Retrieval2.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.001000)
        
        # --- Prepare to start Routine "fixation" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fixation.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        fixationComponents = [fix, MarkEvents_fixation]
        for thisComponent in fixationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fix* updates
            
            # if fix is starting this frame...
            if fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix.frameNStart = frameN  # exact frame index
                fix.tStart = t  # local t and not account for scr refresh
                fix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix.started')
                # update status
                fix.status = STARTED
                fix.setAutoDraw(True)
            
            # if fix is active this frame...
            if fix.status == STARTED:
                # update params
                pass
            
            # if fix is stopping this frame...
            if fix.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fix.tStop = t  # not accounting for scr refresh
                    fix.tStopRefresh = tThisFlipGlobal  # on global time
                    fix.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix.stopped')
                    # update status
                    fix.status = FINISHED
                    fix.setAutoDraw(False)
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('fixation.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "Q2Text" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Q2Text.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        Q2TextComponents = [question_text, MarkEvents_Q2Text]
        for thisComponent in Q2TextComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Q2Text" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > 2-frameTolerance:
                continueRoutine = False
            
            # *question_text* updates
            
            # if question_text is starting this frame...
            if question_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                question_text.frameNStart = frameN  # exact frame index
                question_text.tStart = t  # local t and not account for scr refresh
                question_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(question_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'question_text.started')
                # update status
                question_text.status = STARTED
                question_text.setAutoDraw(True)
            
            # if question_text is active this frame...
            if question_text.status == STARTED:
                # update params
                pass
            
            # if question_text is stopping this frame...
            if question_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > question_text.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    question_text.tStop = t  # not accounting for scr refresh
                    question_text.tStopRefresh = tThisFlipGlobal  # on global time
                    question_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'question_text.stopped')
                    # update status
                    question_text.status = FINISHED
                    question_text.setAutoDraw(False)
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Q2TextComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Q2Text" ---
        for thisComponent in Q2TextComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Q2Text.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "RetrievalImage2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('RetrievalImage2.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from ret_img_c_2
        retrievalImage = None
        
        if retProb < 1.1:
            retType = "old"
            RetrievalBlock2.addData("retType", "old")
            length = len(samePics2)
            if length == 0:
                sys.exit(1)
            retrievalImage = 'images/' + samePics2[retSame2] + '.png'
            retSame2 += 1
        else:
            retType = "new"
            RetrievalBlock2.addData("retType", "new")
            length = len(newPics2)
            if length == 0:
                sys.exit(1)
            retrievalImage = 'images/' + newPics2[retNew2] + '.png'
            retNew2 += 1
        
        # retIMG.setImage(retrievalImage)
        trial_category = "retrieval"
        RetrievalBlock2.addData("retIMG", retrievalImage)
        thisExp.addData("retIMG", retrievalImage)
        retIMG_2.setImage(retrievalImage)
        sound_pop7_2.setSound('POP.wav', secs=0.5, hamming=True)
        sound_pop7_2.setVolume(1.0, log=False)
        sound_pop7_2.seek(0)
        # keep track of which components have finished
        RetrievalImage2Components = [retIMG_2, sound_pop7_2, MarkEvents_Retrieval2]
        for thisComponent in RetrievalImage2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "RetrievalImage2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *retIMG_2* updates
            
            # if retIMG_2 is starting this frame...
            if retIMG_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                retIMG_2.frameNStart = frameN  # exact frame index
                retIMG_2.tStart = t  # local t and not account for scr refresh
                retIMG_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(retIMG_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'retIMG_2.started')
                # update status
                retIMG_2.status = STARTED
                retIMG_2.setAutoDraw(True)
            
            # if retIMG_2 is active this frame...
            if retIMG_2.status == STARTED:
                # update params
                pass
            
            # if retIMG_2 is stopping this frame...
            if retIMG_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > retIMG_2.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    retIMG_2.tStop = t  # not accounting for scr refresh
                    retIMG_2.tStopRefresh = tThisFlipGlobal  # on global time
                    retIMG_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'retIMG_2.stopped')
                    # update status
                    retIMG_2.status = FINISHED
                    retIMG_2.setAutoDraw(False)
            
            # if sound_pop7_2 is starting this frame...
            if sound_pop7_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_pop7_2.frameNStart = frameN  # exact frame index
                sound_pop7_2.tStart = t  # local t and not account for scr refresh
                sound_pop7_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_pop7_2.started', tThisFlipGlobal)
                # update status
                sound_pop7_2.status = STARTED
                sound_pop7_2.play(when=win)  # sync with win flip
            
            # if sound_pop7_2 is stopping this frame...
            if sound_pop7_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_pop7_2.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_pop7_2.tStop = t  # not accounting for scr refresh
                    sound_pop7_2.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_pop7_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_pop7_2.stopped')
                    # update status
                    sound_pop7_2.status = FINISHED
                    sound_pop7_2.stop()
            # update sound_pop7_2 status according to whether it's playing
            if sound_pop7_2.isPlaying:
                sound_pop7_2.status = STARTED
            elif sound_pop7_2.isFinished:
                sound_pop7_2.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in RetrievalImage2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "RetrievalImage2" ---
        for thisComponent in RetrievalImage2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('RetrievalImage2.stopped', globalClock.getTime(format='float'))
        sound_pop7_2.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "Decision2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Decision2.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from dec_code_2
        win.mouseVisible = True
        # setup some python lists for storing info about the Order_Position_Response_2
        Order_Position_Response_2.clicked_name = []
        gotValidClick = False  # until a click is received
        # keep track of which components have finished
        Decision2Components = [decisionQ_3, first_left_2, second_middle_2, third_right_2, First_2, Second_2, Third_2, Order_Position_Response_2, MarkEvents_Retrieval2_Decision2]
        for thisComponent in Decision2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Decision2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *decisionQ_3* updates
            
            # if decisionQ_3 is starting this frame...
            if decisionQ_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                decisionQ_3.frameNStart = frameN  # exact frame index
                decisionQ_3.tStart = t  # local t and not account for scr refresh
                decisionQ_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(decisionQ_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'decisionQ_3.started')
                # update status
                decisionQ_3.status = STARTED
                decisionQ_3.setAutoDraw(True)
            
            # if decisionQ_3 is active this frame...
            if decisionQ_3.status == STARTED:
                # update params
                pass
            
            # *first_left_2* updates
            
            # if first_left_2 is starting this frame...
            if first_left_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                first_left_2.frameNStart = frameN  # exact frame index
                first_left_2.tStart = t  # local t and not account for scr refresh
                first_left_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(first_left_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'first_left_2.started')
                # update status
                first_left_2.status = STARTED
                first_left_2.setAutoDraw(True)
            
            # if first_left_2 is active this frame...
            if first_left_2.status == STARTED:
                # update params
                pass
            
            # *second_middle_2* updates
            
            # if second_middle_2 is starting this frame...
            if second_middle_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                second_middle_2.frameNStart = frameN  # exact frame index
                second_middle_2.tStart = t  # local t and not account for scr refresh
                second_middle_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(second_middle_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'second_middle_2.started')
                # update status
                second_middle_2.status = STARTED
                second_middle_2.setAutoDraw(True)
            
            # if second_middle_2 is active this frame...
            if second_middle_2.status == STARTED:
                # update params
                pass
            
            # *third_right_2* updates
            
            # if third_right_2 is starting this frame...
            if third_right_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                third_right_2.frameNStart = frameN  # exact frame index
                third_right_2.tStart = t  # local t and not account for scr refresh
                third_right_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(third_right_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'third_right_2.started')
                # update status
                third_right_2.status = STARTED
                third_right_2.setAutoDraw(True)
            
            # if third_right_2 is active this frame...
            if third_right_2.status == STARTED:
                # update params
                pass
            
            # *First_2* updates
            
            # if First_2 is starting this frame...
            if First_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                First_2.frameNStart = frameN  # exact frame index
                First_2.tStart = t  # local t and not account for scr refresh
                First_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(First_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'First_2.started')
                # update status
                First_2.status = STARTED
                First_2.setAutoDraw(True)
            
            # if First_2 is active this frame...
            if First_2.status == STARTED:
                # update params
                pass
            
            # *Second_2* updates
            
            # if Second_2 is starting this frame...
            if Second_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Second_2.frameNStart = frameN  # exact frame index
                Second_2.tStart = t  # local t and not account for scr refresh
                Second_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Second_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Second_2.started')
                # update status
                Second_2.status = STARTED
                Second_2.setAutoDraw(True)
            
            # if Second_2 is active this frame...
            if Second_2.status == STARTED:
                # update params
                pass
            
            # *Third_2* updates
            
            # if Third_2 is starting this frame...
            if Third_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Third_2.frameNStart = frameN  # exact frame index
                Third_2.tStart = t  # local t and not account for scr refresh
                Third_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Third_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Third_2.started')
                # update status
                Third_2.status = STARTED
                Third_2.setAutoDraw(True)
            
            # if Third_2 is active this frame...
            if Third_2.status == STARTED:
                # update params
                pass
            # *Order_Position_Response_2* updates
            
            # if Order_Position_Response_2 is starting this frame...
            if Order_Position_Response_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Order_Position_Response_2.frameNStart = frameN  # exact frame index
                Order_Position_Response_2.tStart = t  # local t and not account for scr refresh
                Order_Position_Response_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Order_Position_Response_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('Order_Position_Response_2.started', t)
                # update status
                Order_Position_Response_2.status = STARTED
                Order_Position_Response_2.mouseClock.reset()
                prevButtonState = Order_Position_Response_2.getPressed()  # if button is down already this ISN'T a new click
            if Order_Position_Response_2.status == STARTED:  # only update if started and not finished!
                buttons = Order_Position_Response_2.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([first_left,second_middle,third_right], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(Order_Position_Response_2):
                                gotValidClick = True
                                Order_Position_Response_2.clicked_name.append(obj.name)
                        if gotValidClick:  
                            continueRoutine = False  # end routine on response
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Decision2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Decision2" ---
        for thisComponent in Decision2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Decision2.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from dec_code_2
        retTrials2 = retTrials2 + 1
        TRIAL_INDEX = TRIAL_INDEX + 1
        #psychopy output
        RetrievalBlock2.addData("retTrials", str(retTrials2))
        RetrievalBlock2.addData("TRIAL_INDEX", str(TRIAL_INDEX))
        RetrievalBlock2.addData("retIMG", str(retrievalImage))
        RetrievalBlock2.addData('trialCategory', str(trial_category))
        
        #for EDF output
        el_tracker.sendMessage('!V TRIAL_VAR retTrials %d' % retTrials2)
        #el_tracker.sendMessage('!V TRIAL_VAR orderResponse %s' % orderResponse)
        el_tracker.sendMessage('!V TRIAL_VAR orderPosition %s' % Order_Position_Response_2.clicked_name)
        
        
        win.mouseVisible = False
        # store data for RetrievalBlock2 (TrialHandler)
        x, y = Order_Position_Response_2.getPos()
        buttons = Order_Position_Response_2.getPressed()
        if sum(buttons):
            # check if the mouse was inside our 'clickable' objects
            gotValidClick = False
            clickableList = environmenttools.getFromNames([first_left,second_middle,third_right], namespace=locals())
            for obj in clickableList:
                # is this object clicked on?
                if obj.contains(Order_Position_Response_2):
                    gotValidClick = True
                    Order_Position_Response_2.clicked_name.append(obj.name)
        RetrievalBlock2.addData('Order_Position_Response_2.x', x)
        RetrievalBlock2.addData('Order_Position_Response_2.y', y)
        RetrievalBlock2.addData('Order_Position_Response_2.leftButton', buttons[0])
        RetrievalBlock2.addData('Order_Position_Response_2.midButton', buttons[1])
        RetrievalBlock2.addData('Order_Position_Response_2.rightButton', buttons[2])
        if len(Order_Position_Response_2.clicked_name):
            RetrievalBlock2.addData('Order_Position_Response_2.clicked_name', Order_Position_Response_2.clicked_name[0])
        # the Routine "Decision2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "stopRecording" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('stopRecording.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        stopRecordingComponents = [StopRecord]
        for thisComponent in stopRecordingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "stopRecording" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.001:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in stopRecordingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "stopRecording" ---
        for thisComponent in stopRecordingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('stopRecording.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.001000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed numRetrieval2 repeats of 'RetrievalBlock2'
    
    
    # --- Prepare to start Routine "EndScreen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('EndScreen.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    EndScreenComponents = [end]
    for thisComponent in EndScreenComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "EndScreen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end* updates
        
        # if end is starting this frame...
        if end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end.frameNStart = frameN  # exact frame index
            end.tStart = t  # local t and not account for scr refresh
            end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end.started')
            # update status
            end.status = STARTED
            end.setAutoDraw(True)
        
        # if end is active this frame...
        if end.status == STARTED:
            # update params
            pass
        
        # if end is stopping this frame...
        if end.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                end.tStop = t  # not accounting for scr refresh
                end.tStopRefresh = tThisFlipGlobal  # on global time
                end.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end.stopped')
                # update status
                end.status = FINISHED
                end.setAutoDraw(False)
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in EndScreenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "EndScreen" ---
    for thisComponent in EndScreenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('EndScreen.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
